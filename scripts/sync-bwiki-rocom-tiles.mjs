#!/usr/bin/env node

import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, "..");
const DEFAULT_OUT_DIR = path.join(REPO_ROOT, ".tmp-bwiki-rocom", "tiles");
const DEFAULT_URL_TEMPLATE =
  "https://wiki-dev-patch-oss.oss-cn-hangzhou.aliyuncs.com/res/lkwg/map-3.0/{z}/tile-{x}_{y}.png";

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

function log(message) {
  console.log(`[bwiki-tiles] ${message}`);
}

main().catch((error) => {
  console.error(`[bwiki-tiles] ${error.stack ?? error.message}`);
  process.exitCode = 1;
});
