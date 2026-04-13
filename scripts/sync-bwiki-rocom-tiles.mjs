#!/usr/bin/env node

import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { createWriteStream } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createDeflate, inflateSync } from "node:zlib";
import { once } from "node:events";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, "..");
const DEFAULT_OUT_DIR = path.join(REPO_ROOT, ".tmp-bwiki-rocom", "tiles");
const DEFAULT_URL_TEMPLATE =
  "https://wiki-dev-patch-oss.oss-cn-hangzhou.aliyuncs.com/res/lkwg/map-3.0/{z}/tile-{x}_{y}.png";

const PNG_SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const CRC_TABLE = buildCrcTable();

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(options.outDir ?? DEFAULT_OUT_DIR);

  await mkdir(outDir, { recursive: true });

  const zooms = [];
  for (let zoom = options.minZoom; zoom <= options.maxZoom; zoom += 1) {
    zooms.push(zoom);
  }

  const manifest = {
    fetchedAt: new Date().toISOString(),
    source: {
      urlTemplate: options.urlTemplate,
    },
    options: {
      minZoom: options.minZoom,
      maxZoom: options.maxZoom,
      scanLimit: options.scanLimit,
      concurrency: options.concurrency,
      skipStitch: options.skipStitch,
      force: options.force,
      probeOnly: options.probeOnly,
    },
    zooms: [],
  };

  for (const zoom of zooms) {
    log(`discovering range for z=${zoom}`);
    const range = await discoverTileRange(zoom, options);
    const tileCount = (range.maxX - range.minX + 1) * (range.maxY - range.minY + 1);
    const zoomEntry = {
      zoom,
      range,
      tileCount,
      tilesDir: `tiles/z${zoom}`,
    };
    manifest.zooms.push(zoomEntry);

    log(
      `z=${zoom} range x=${range.minX}..${range.maxX} y=${range.minY}..${range.maxY} expected=${tileCount}`,
    );

    if (options.probeOnly) {
      continue;
    }

    const downloadStats = await downloadZoomTiles(zoom, range, outDir, options);
    zoomEntry.download = downloadStats;

    if (!options.skipStitch) {
      const stitched = await stitchZoomTiles(zoom, range, outDir);
      zoomEntry.stitched = stitched;
      log(
        `z=${zoom} stitched ${stitched.width}x${stitched.height} -> ${stitched.imagePath}`,
      );
    }
  }

  await writeJson(path.join(outDir, "manifest.json"), manifest);
  log(`manifest written to ${path.join(outDir, "manifest.json")}`);
}

function parseArgs(argv) {
  const options = {
    outDir: null,
    minZoom: 4,
    maxZoom: 8,
    scanLimit: 64,
    concurrency: 16,
    skipStitch: false,
    force: false,
    probeOnly: false,
    urlTemplate: DEFAULT_URL_TEMPLATE,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === "--out-dir") {
      options.outDir = requireValue(argv, ++index, arg);
      continue;
    }
    if (arg === "--min-zoom") {
      options.minZoom = parseInteger(requireValue(argv, ++index, arg), arg);
      continue;
    }
    if (arg === "--max-zoom") {
      options.maxZoom = parseInteger(requireValue(argv, ++index, arg), arg);
      continue;
    }
    if (arg === "--scan-limit") {
      options.scanLimit = parseInteger(requireValue(argv, ++index, arg), arg);
      continue;
    }
    if (arg === "--concurrency") {
      options.concurrency = parseInteger(requireValue(argv, ++index, arg), arg);
      continue;
    }
    if (arg === "--url-template") {
      options.urlTemplate = requireValue(argv, ++index, arg);
      continue;
    }
    if (arg === "--skip-stitch") {
      options.skipStitch = true;
      continue;
    }
    if (arg === "--force") {
      options.force = true;
      continue;
    }
    if (arg === "--probe-only") {
      options.probeOnly = true;
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }

    throw new Error(`unknown argument: ${arg}`);
  }

  if (options.minZoom > options.maxZoom) {
    throw new Error("--min-zoom must be <= --max-zoom");
  }
  if (options.scanLimit < 1) {
    throw new Error("--scan-limit must be >= 1");
  }
  if (options.concurrency < 1) {
    throw new Error("--concurrency must be >= 1");
  }

  return options;
}

function printHelp() {
  console.log(`Usage:
  node scripts/sync-bwiki-rocom-tiles.mjs [options]

Options:
  --out-dir <dir>        Output directory, default .tmp-bwiki-rocom/tiles
  --min-zoom <n>         Minimum zoom, default 4
  --max-zoom <n>         Maximum zoom, default 8
  --scan-limit <n>       Axis probe half-range, default 64
  --concurrency <n>      Concurrent HTTP requests, default 16
  --url-template <url>   Tile URL template with {z} {x} {y}
  --skip-stitch          Download tiles but do not stitch PNGs
  --force                Re-download tiles even if local files exist
  --probe-only           Only discover ranges, do not download tiles
  --help, -h             Show this help text
`);
}

function requireValue(argv, index, flag) {
  const value = argv[index];
  if (value == null) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
}

function parseInteger(value, flag) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${flag} requires an integer`);
  }
  return parsed;
}

async function discoverTileRange(zoom, options) {
  const xHits = await probeAxis(zoom, "x", options);
  const yHits = await probeAxis(zoom, "y", options);

  if (xHits.length === 0 || yHits.length === 0) {
    throw new Error(`no valid tiles discovered for z=${zoom}`);
  }

  return {
    minX: xHits[0],
    maxX: xHits[xHits.length - 1],
    minY: yHits[0],
    maxY: yHits[yHits.length - 1],
    xHits,
    yHits,
  };
}

async function probeAxis(zoom, axis, options) {
  const values = [];
  for (let coordinate = -options.scanLimit; coordinate <= options.scanLimit; coordinate += 1) {
    values.push(coordinate);
  }

  const hits = [];
  await runPool(values, options.concurrency, async (coordinate) => {
    const x = axis === "x" ? coordinate : 0;
    const y = axis === "y" ? coordinate : 0;
    const ok = await probeTile(urlFor(options.urlTemplate, zoom, x, y));
    if (ok) {
      hits.push(coordinate);
    }
  });

  return hits.sort((left, right) => left - right);
}

async function probeTile(url) {
  try {
    const head = await fetch(url, {
      method: "HEAD",
      headers: {
        "user-agent": "game-map-tracker-rs-bwiki-tiles/1.0",
      },
    });
    if (head.ok) {
      return true;
    }
    if (head.status !== 405) {
      return false;
    }
  } catch {}

  try {
    const response = await fetch(url, {
      headers: {
        range: "bytes=0-0",
        "user-agent": "game-map-tracker-rs-bwiki-tiles/1.0",
      },
    });
    return response.ok;
  } catch {
    return false;
  }
}

async function downloadZoomTiles(zoom, range, outDir, options) {
  const zoomDir = path.join(outDir, "tiles", `z${zoom}`);
  await mkdir(zoomDir, { recursive: true });

  const jobs = [];
  for (let y = range.minY; y <= range.maxY; y += 1) {
    for (let x = range.minX; x <= range.maxX; x += 1) {
      jobs.push({ x, y });
    }
  }

  const stats = {
    expected: jobs.length,
    downloaded: 0,
    skippedExisting: 0,
    failed: [],
  };

  await runPool(jobs, options.concurrency, async ({ x, y }) => {
    const destination = tilePath(outDir, zoom, x, y);
    if (!options.force && (await fileExists(destination))) {
      stats.skippedExisting += 1;
      return;
    }

    const response = await fetch(urlFor(options.urlTemplate, zoom, x, y), {
      headers: {
        "user-agent": "game-map-tracker-rs-bwiki-tiles/1.0",
      },
    });

    if (!response.ok) {
      stats.failed.push({
        x,
        y,
        status: response.status,
        statusText: response.statusText,
      });
      return;
    }

    const bytes = Buffer.from(await response.arrayBuffer());
    await writeFile(destination, bytes);
    stats.downloaded += 1;
  });

  stats.failed.sort((left, right) => left.y - right.y || left.x - right.x);
  return stats;
}

async function stitchZoomTiles(zoom, range, outDir) {
  const firstTile = await findFirstTile(outDir, zoom, range);
  if (!firstTile) {
    throw new Error(`no tiles downloaded for z=${zoom}`);
  }

  const sample = decodePng(await readFile(firstTile));
  const tileWidth = sample.width;
  const tileHeight = sample.height;
  const columns = range.maxX - range.minX + 1;
  const rows = range.maxY - range.minY + 1;
  const width = columns * tileWidth;
  const height = rows * tileHeight;

  const imagesDir = path.join(outDir, "stitched");
  await mkdir(imagesDir, { recursive: true });
  const imagePath = path.join(imagesDir, `z${zoom}.png`);

  const output = createWriteStream(imagePath);
  await writeChunked(output, PNG_SIGNATURE);
  await writePngChunk(output, "IHDR", buildIhdr(width, height));

  const deflater = createDeflate({ level: 9 });
  const idatWriter = (async () => {
    for await (const chunk of deflater) {
      await writePngChunk(output, "IDAT", chunk);
    }
  })();

  for (let tileY = range.minY; tileY <= range.maxY; tileY += 1) {
    const rowTiles = await loadTileRow(outDir, zoom, range, tileY, tileWidth, tileHeight);
    for (let pixelRow = 0; pixelRow < tileHeight; pixelRow += 1) {
      const scanline = Buffer.alloc(1 + width * 4);
      scanline[0] = 0;
      let offset = 1;
      for (const tile of rowTiles) {
        const start = pixelRow * tileWidth * 4;
        const end = start + tileWidth * 4;
        tile.copy(scanline, offset, start, end);
        offset += tileWidth * 4;
      }
      await writeChunked(deflater, scanline);
    }
  }

  deflater.end();
  await idatWriter;
  await writePngChunk(output, "IEND", Buffer.alloc(0));
  output.end();
  await once(output, "finish");

  return {
    imagePath: path.relative(outDir, imagePath).replaceAll("\\", "/"),
    width,
    height,
    tileWidth,
    tileHeight,
  };
}

async function loadTileRow(outDir, zoom, range, tileY, tileWidth, tileHeight) {
  const row = [];
  for (let tileX = range.minX; tileX <= range.maxX; tileX += 1) {
    const source = tilePath(outDir, zoom, tileX, tileY);
    if (!(await fileExists(source))) {
      row.push(Buffer.alloc(tileWidth * tileHeight * 4));
      continue;
    }
    const tile = decodePng(await readFile(source));
    if (tile.width !== tileWidth || tile.height !== tileHeight) {
      throw new Error(`tile size mismatch for ${source}`);
    }
    row.push(tile.rgba);
  }
  return row;
}

async function findFirstTile(outDir, zoom, range) {
  for (let y = range.minY; y <= range.maxY; y += 1) {
    for (let x = range.minX; x <= range.maxX; x += 1) {
      const source = tilePath(outDir, zoom, x, y);
      if (await fileExists(source)) {
        return source;
      }
    }
  }
  return null;
}

function tilePath(outDir, zoom, x, y) {
  return path.join(outDir, "tiles", `z${zoom}`, `tile-${x}_${y}.png`);
}

function urlFor(template, zoom, x, y) {
  return template
    .replaceAll("{z}", String(zoom))
    .replaceAll("{x}", String(x))
    .replaceAll("{y}", String(y));
}

async function runPool(items, limit, worker) {
  const executing = new Set();
  for (const item of items) {
    const task = Promise.resolve().then(() => worker(item));
    executing.add(task);
    task.finally(() => {
      executing.delete(task);
    });
    if (executing.size >= limit) {
      await Promise.race(executing);
    }
  }
  await Promise.all(executing);
}

async function fileExists(filePath) {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function writeJson(filePath, value) {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function decodePng(buffer) {
  if (!buffer.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)) {
    throw new Error("invalid png signature");
  }

  let offset = PNG_SIGNATURE.length;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = 0;
  let interlaceMethod = 0;
  let palette = null;
  let transparency = null;
  const idatChunks = [];

  while (offset < buffer.length) {
    const length = buffer.readUInt32BE(offset);
    offset += 4;
    const type = buffer.subarray(offset, offset + 4).toString("ascii");
    offset += 4;
    const data = buffer.subarray(offset, offset + length);
    offset += length;
    offset += 4;

    if (type === "IHDR") {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      bitDepth = data[8];
      colorType = data[9];
      interlaceMethod = data[12];
    } else if (type === "PLTE") {
      palette = Buffer.from(data);
    } else if (type === "tRNS") {
      transparency = Buffer.from(data);
    } else if (type === "IDAT") {
      idatChunks.push(data);
    } else if (type === "IEND") {
      break;
    }
  }

  if (bitDepth !== 8) {
    throw new Error(`unsupported png bit depth ${bitDepth}`);
  }
  if (interlaceMethod !== 0) {
    throw new Error("interlaced png is not supported");
  }

  const bytesPerPixel = colorBytesPerPixel(colorType);
  const scanlineLength = width * bytesPerPixel;
  const inflated = inflateSync(Buffer.concat(idatChunks));
  const raw = Buffer.alloc(width * height * 4);

  let sourceOffset = 0;
  const previous = Buffer.alloc(scanlineLength);
  const current = Buffer.alloc(scanlineLength);

  for (let row = 0; row < height; row += 1) {
    const filter = inflated[sourceOffset];
    sourceOffset += 1;

    for (let index = 0; index < scanlineLength; index += 1) {
      const value = inflated[sourceOffset];
      sourceOffset += 1;

      const left = index >= bytesPerPixel ? current[index - bytesPerPixel] : 0;
      const up = previous[index];
      const upLeft = index >= bytesPerPixel ? previous[index - bytesPerPixel] : 0;

      current[index] = applyFilter(filter, value, left, up, upLeft);
    }

    writeRgbaRow(raw, row, width, colorType, current, palette, transparency);
    current.copy(previous);
  }

  return { width, height, rgba: raw };
}

function colorBytesPerPixel(colorType) {
  switch (colorType) {
    case 0:
      return 1;
    case 2:
      return 3;
    case 3:
      return 1;
    case 4:
      return 2;
    case 6:
      return 4;
    default:
      throw new Error(`unsupported png color type ${colorType}`);
  }
}

function applyFilter(filter, value, left, up, upLeft) {
  switch (filter) {
    case 0:
      return value;
    case 1:
      return (value + left) & 255;
    case 2:
      return (value + up) & 255;
    case 3:
      return (value + Math.floor((left + up) / 2)) & 255;
    case 4:
      return (value + paeth(left, up, upLeft)) & 255;
    default:
      throw new Error(`unsupported png filter ${filter}`);
  }
}

function paeth(left, up, upLeft) {
  const predictor = left + up - upLeft;
  const pLeft = Math.abs(predictor - left);
  const pUp = Math.abs(predictor - up);
  const pUpLeft = Math.abs(predictor - upLeft);

  if (pLeft <= pUp && pLeft <= pUpLeft) {
    return left;
  }
  if (pUp <= pUpLeft) {
    return up;
  }
  return upLeft;
}

function writeRgbaRow(target, row, width, colorType, source, palette, transparency) {
  let sourceOffset = 0;
  let targetOffset = row * width * 4;

  for (let column = 0; column < width; column += 1) {
    if (colorType === 6) {
      target[targetOffset++] = source[sourceOffset++];
      target[targetOffset++] = source[sourceOffset++];
      target[targetOffset++] = source[sourceOffset++];
      target[targetOffset++] = source[sourceOffset++];
      continue;
    }

    if (colorType === 2) {
      const red = source[sourceOffset++];
      const green = source[sourceOffset++];
      const blue = source[sourceOffset++];
      target[targetOffset++] = red;
      target[targetOffset++] = green;
      target[targetOffset++] = blue;
      target[targetOffset++] =
        transparency && transparency.length >= 6 &&
        red === transparency[1] &&
        green === transparency[3] &&
        blue === transparency[5]
          ? 0
          : 255;
      continue;
    }

    if (colorType === 4) {
      const gray = source[sourceOffset++];
      const alpha = source[sourceOffset++];
      target[targetOffset++] = gray;
      target[targetOffset++] = gray;
      target[targetOffset++] = gray;
      target[targetOffset++] = alpha;
      continue;
    }

    if (colorType === 0) {
      const gray = source[sourceOffset++];
      target[targetOffset++] = gray;
      target[targetOffset++] = gray;
      target[targetOffset++] = gray;
      target[targetOffset++] =
        transparency && transparency.length >= 2 && gray === transparency[1] ? 0 : 255;
      continue;
    }

    if (colorType === 3) {
      const index = source[sourceOffset++];
      const paletteOffset = index * 3;
      target[targetOffset++] = palette?.[paletteOffset] ?? 0;
      target[targetOffset++] = palette?.[paletteOffset + 1] ?? 0;
      target[targetOffset++] = palette?.[paletteOffset + 2] ?? 0;
      target[targetOffset++] = transparency?.[index] ?? 255;
      continue;
    }
  }
}

function buildIhdr(width, height) {
  const chunk = Buffer.alloc(13);
  chunk.writeUInt32BE(width, 0);
  chunk.writeUInt32BE(height, 4);
  chunk[8] = 8;
  chunk[9] = 6;
  chunk[10] = 0;
  chunk[11] = 0;
  chunk[12] = 0;
  return chunk;
}

async function writePngChunk(stream, type, data) {
  const typeBuffer = Buffer.from(type, "ascii");
  const lengthBuffer = Buffer.alloc(4);
  lengthBuffer.writeUInt32BE(data.length, 0);
  const crcBuffer = Buffer.alloc(4);
  crcBuffer.writeUInt32BE(crc32(Buffer.concat([typeBuffer, data])), 0);

  await writeChunked(stream, lengthBuffer);
  await writeChunked(stream, typeBuffer);
  if (data.length > 0) {
    await writeChunked(stream, data);
  }
  await writeChunked(stream, crcBuffer);
}

async function writeChunked(stream, chunk) {
  if (!stream.write(chunk)) {
    await once(stream, "drain");
  }
}

function buildCrcTable() {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      if ((value & 1) !== 0) {
        value = 0xedb88320 ^ (value >>> 1);
      } else {
        value >>>= 1;
      }
    }
    table[index] = value >>> 0;
  }
  return table;
}

function crc32(buffer) {
  let value = 0xffffffff;
  for (const byte of buffer) {
    value = CRC_TABLE[(value ^ byte) & 255] ^ (value >>> 8);
  }
  return (value ^ 0xffffffff) >>> 0;
}

function log(message) {
  console.log(`[bwiki-tiles] ${message}`);
}

main().catch((error) => {
  console.error(`[bwiki-tiles] ${error.stack ?? error.message}`);
  process.exitCode = 1;
});
