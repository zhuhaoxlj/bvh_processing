/** Lightweight IndexedDB blob cache for MuJoCo WASM / G1 MJB second-visit hits. */

const DB_NAME = "g1-policy-preview-cache";
const DB_VERSION = 1;
const STORE = "blobs";

export type CacheSource = "idb" | "network" | "memory" | "none";

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("IndexedDB unavailable"));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB open failed"));
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    request.onsuccess = () => resolve(request.result);
  });
}

export async function cacheGet(key: string): Promise<ArrayBuffer | null> {
  try {
    const db = await openDb();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, "readonly");
      const req = tx.objectStore(STORE).get(key);
      req.onerror = () => reject(req.error ?? new Error("cache get failed"));
      req.onsuccess = () => {
        const value = req.result;
        if (value instanceof ArrayBuffer) resolve(value);
        else if (value instanceof Uint8Array) {
          const copy = new Uint8Array(value.byteLength);
          copy.set(value);
          resolve(copy.buffer);
        } else resolve(null);
      };
      tx.oncomplete = () => db.close();
    });
  } catch {
    return null;
  }
}

export async function cacheSet(key: string, data: ArrayBuffer): Promise<void> {
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      // Store a copy so callers can safely detach/transfer the original buffer.
      tx.objectStore(STORE).put(data.slice(0), key);
      tx.oncomplete = () => {
        db.close();
        resolve();
      };
      tx.onerror = () => reject(tx.error ?? new Error("cache set failed"));
    });
  } catch {
    // Cache is best-effort; ignore quota / private-mode failures.
  }
}

/** Fetch with IndexedDB hit/miss. Always returns a fresh ArrayBuffer copy of the payload. */
export async function fetchCached(
  key: string,
  url: string,
): Promise<{ buffer: ArrayBuffer; source: CacheSource }> {
  const cached = await cacheGet(key);
  if (cached && cached.byteLength > 0) {
    return { buffer: cached.slice(0), source: "idb" };
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`无法下载 ${url} (${response.status})`);
  const buffer = await response.arrayBuffer();
  void cacheSet(key, buffer);
  return { buffer, source: "network" };
}

export function assetUrl(relativePath: string): string {
  const base = import.meta.env.BASE_URL || "/";
  const cleaned = relativePath.replace(/^\//, "");
  if (base.endsWith("/")) return `${base}${cleaned}`;
  return `${base}/${cleaned}`;
}
