#!/usr/bin/env node

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, "..");
const DEFAULT_OUT_DIR = path.join(REPO_ROOT, ".tmp-bwiki-rocom");
const WIKI_BASE_URL = "https://wiki.biligame.com/rocom";
const API_URL = `${WIKI_BASE_URL}/api.php`;

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(options.outDir ?? DEFAULT_OUT_DIR);

  log(`output directory: ${outDir}`);
  log("fetching type catalog");
  const types = await fetchTypeCatalog();

  log("fetching expanded point payload");
  const pointsByType = await fetchPointsByType();

  const normalizedTypes = normalizeTypes(types, pointsByType);
  const flatPoints = buildFlatPoints(normalizedTypes, pointsByType);
  const summary = buildSummary(normalizedTypes, pointsByType, flatPoints);

  let iconFailures = [];
  if (!options.skipIcons) {
    log("downloading icons");
    iconFailures = await downloadIcons(normalizedTypes, outDir);
  }

  await mkdir(outDir, { recursive: true });
  await writeJson(path.join(outDir, "types.json"), normalizedTypes);
  await writeJson(path.join(outDir, "points-by-type.json"), pointsByType);
  await writeJson(path.join(outDir, "flat-points.json"), flatPoints);
  await writeJson(path.join(outDir, "summary.json"), {
    ...summary,
    iconDownloadFailures: iconFailures,
  });
  await writeJson(path.join(outDir, "manifest.json"), {
    fetchedAt: new Date().toISOString(),
    source: {
      wikiBaseUrl: WIKI_BASE_URL,
      typePage: "Data:Mapnew/type/json",
      pointPage: "Data:Mapnew/point.json",
    },
    outputs: {
      types: "types.json",
      pointsByType: "points-by-type.json",
      flatPoints: "flat-points.json",
      summary: "summary.json",
      iconsDir: options.skipIcons ? null : "icons",
    },
    counts: {
      typeCount: summary.typeCount,
      pointTypeCount: summary.pointTypeCount,
      nonEmptyPointTypeCount: summary.nonEmptyPointTypeCount,
      catalogMatchedPointTypeCount: summary.catalogMatchedPointTypeCount,
      catalogMatchedNonEmptyTypeCount: summary.catalogMatchedNonEmptyTypeCount,
      unknownPointTypeCount: summary.unknownPointTypes.length,
      pointCount: summary.pointCount,
    },
    notes: [
      "point.lat/lng are raw BWiki leaflet coordinates",
      "they are not converted into this repository's display_map pixel space",
    ],
  });

  log(
    `done: ${summary.typeCount} types, ${summary.pointCount} points, ${summary.nonEmptyPointTypeCount} non-empty point types, ${summary.unknownPointTypes.length} unknown point types`,
  );
}

function parseArgs(argv) {
  const options = {
    outDir: null,
    skipIcons: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === "--out-dir") {
      const value = argv[index + 1];
      if (!value) {
        throw new Error("--out-dir requires a value");
      }
      options.outDir = value;
      index += 1;
      continue;
    }

    if (arg === "--skip-icons") {
      options.skipIcons = true;
      continue;
    }

    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }

    throw new Error(`unknown argument: ${arg}`);
  }

  return options;
}

function printHelp() {
  console.log(`Usage:
  node scripts/sync-bwiki-rocom.mjs [--out-dir <dir>] [--skip-icons]

Options:
  --out-dir <dir>   Write exported files into a custom directory
  --skip-icons      Skip downloading icon files
  --help, -h        Show this help message
`);
}

function buildApiUrl(params) {
  const url = new URL(API_URL);
  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, value);
  }
  return url.toString();
}

async function fetchTypeCatalog() {
  const payload = await fetchJson(
    buildApiUrl({
      action: "parse",
      page: "Data:Mapnew/type/json",
      prop: "text",
      format: "json",
      formatversion: "2",
    }),
  );

  const rawText = htmlToText(payload.parse.text);
  const parsed = JSON.parse(rawText);
  return parsed.data;
}

async function fetchPointsByType() {
  const payload = await fetchJson(
    buildApiUrl({
      action: "parse",
      page: "Data:Mapnew/point.json",
      prop: "text",
      format: "json",
      formatversion: "2",
    }),
  );

  let rawText = htmlToText(payload.parse.text);
  rawText = rawText.replace(/:Data:[\s\S]{0,30}?\/json/g, ":[]");

  const parsed = Function(`"use strict"; return (${rawText});`)();
  const normalized = {};

  for (const [key, value] of Object.entries(parsed)) {
    if (!Array.isArray(value)) {
      continue;
    }
    normalized[key] = value;
  }

  return normalized;
}

function normalizeTypes(types, pointsByType) {
  return types
    .map((entry) => {
      const markType = Number(entry.markType);
      const key = String(markType);
      const points = pointsByType[key] ?? [];
      const iconUrl = normalizeIconUrl(entry.icon);

      return {
        category: entry.type,
        markType,
        name: entry.markTypeName,
        description: entry.desc ?? "",
        defaultShow: entry.defaultShow ?? "",
        className: entry.class ?? "",
        collectible: entry.collectible ?? "",
        geojson: entry.geojson ?? "",
        iconUrl,
        pointDataTitle: `Data:Mapnew/type/${markType}/json`,
        declaredLength: toInteger(entry.length),
        pointCount: points.length,
        iconLocalPath: `icons/${markType}${iconExtension(iconUrl)}`,
      };
    })
    .sort((left, right) => left.markType - right.markType);
}

function buildFlatPoints(types, pointsByType) {
  const typeByMarkType = new Map(types.map((item) => [item.markType, item]));
  const flatPoints = [];

  for (const [markTypeKey, points] of Object.entries(pointsByType)) {
    const markType = Number(markTypeKey);
    const type = typeByMarkType.get(markType);

    for (const point of points) {
      flatPoints.push({
        typeKnown: Boolean(type),
        category: type?.category ?? "",
        markType,
        typeName: type?.name ?? "",
        iconUrl: type?.iconUrl ?? "",
        id: point.id,
        title: point.title ?? "",
        layer: point.layer ?? "",
        uid: point.uid ?? "",
        time: point.time ?? null,
        version: point.version ?? null,
        point: {
          lat: point.point?.lat ?? null,
          lng: point.point?.lng ?? null,
        },
      });
    }
  }

  return flatPoints.sort((left, right) => {
    if (left.markType !== right.markType) {
      return left.markType - right.markType;
    }
    return String(left.id).localeCompare(String(right.id));
  });
}

function buildSummary(types, pointsByType, flatPoints) {
  const categorySummary = new Map();
  const pointTypeKeys = Object.keys(pointsByType);
  const nonEmptyPointTypeKeys = pointTypeKeys.filter((key) => (pointsByType[key] ?? []).length > 0);
  const typeSet = new Set(types.map((type) => String(type.markType)));
  const unknownPointTypes = nonEmptyPointTypeKeys
    .filter((key) => !typeSet.has(key))
    .map((key) => ({
      markType: Number(key),
      pointCount: pointsByType[key].length,
    }))
    .sort((left, right) => left.markType - right.markType);
  let catalogMatchedPointTypeCount = 0;
  let catalogMatchedNonEmptyTypeCount = 0;

  for (const type of types) {
    const group = categorySummary.get(type.category) ?? {
      category: type.category,
      typeCount: 0,
      catalogMatchedNonEmptyTypeCount: 0,
      pointCount: 0,
    };

    group.typeCount += 1;
    group.pointCount += type.pointCount;
    if (Object.prototype.hasOwnProperty.call(pointsByType, String(type.markType))) {
      catalogMatchedPointTypeCount += 1;
    }
    if (type.pointCount > 0) {
      group.catalogMatchedNonEmptyTypeCount += 1;
      catalogMatchedNonEmptyTypeCount += 1;
    }

    categorySummary.set(type.category, group);
  }

  return {
    typeCount: types.length,
    pointTypeCount: pointTypeKeys.length,
    nonEmptyPointTypeCount: nonEmptyPointTypeKeys.length,
    catalogMatchedPointTypeCount,
    catalogMatchedNonEmptyTypeCount,
    unknownPointTypes,
    pointCount: flatPoints.length,
    categories: Array.from(categorySummary.values()).sort((left, right) =>
      left.category.localeCompare(right.category),
    ),
  };
}

async function downloadIcons(types, outDir) {
  const iconsDir = path.join(outDir, "icons");
  await mkdir(iconsDir, { recursive: true });

  const failures = [];
  await runPool(types, 8, async (type) => {
    if (!type.iconUrl) {
      failures.push({
        markType: type.markType,
        reason: "missing icon url",
      });
      return;
    }

    const response = await fetch(type.iconUrl, {
      headers: {
        "user-agent": "game-map-tracker-rs-bwiki-sync/1.0",
      },
    });

    if (!response.ok) {
      failures.push({
        markType: type.markType,
        reason: `icon download failed: ${response.status} ${response.statusText}`,
      });
      return;
    }

    const bytes = Buffer.from(await response.arrayBuffer());
    const target = path.join(iconsDir, `${type.markType}${iconExtension(type.iconUrl)}`);
    await writeFile(target, bytes);
  });

  return failures.sort((left, right) => left.markType - right.markType);
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

function htmlToText(html) {
  return decodeHtmlEntities(html.replace(/<[^>]*>/g, "")).trim();
}

function decodeHtmlEntities(value) {
  return value
    .replace(/&quot;/g, '"')
    .replace(/&#34;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&apos;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&");
}

function normalizeIconUrl(value) {
  if (!value) {
    return "";
  }

  const trimmed = String(value).trim();
  const match = trimmed.match(/https?:\/\/[^\s"]+/i);
  return match ? match[0] : trimmed;
}

function iconExtension(iconUrl) {
  try {
    const extension = path.extname(new URL(iconUrl).pathname).toLowerCase();
    return extension || ".png";
  } catch {
    return ".png";
  }
}

function toInteger(value) {
  const parsed = Number.parseInt(String(value), 10);
  return Number.isFinite(parsed) ? parsed : null;
}

async function fetchJson(url) {
  const response = await fetch(url, {
    headers: {
      "user-agent": "game-map-tracker-rs-bwiki-sync/1.0",
    },
  });

  if (!response.ok) {
    throw new Error(`request failed: ${response.status} ${response.statusText} for ${url}`);
  }

  return response.json();
}

async function writeJson(filePath, value) {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function log(message) {
  console.log(`[bwiki-sync] ${message}`);
}

main().catch((error) => {
  console.error(`[bwiki-sync] ${error.stack ?? error.message}`);
  process.exitCode = 1;
});
