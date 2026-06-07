/**
 * Simple in-memory LRU cache with TTL.
 * Used to short-circuit repeated /api/scrape, /api/scrape-size, and
 * /api/recommend-size calls for the same product URL.
 */

class SimpleLRU {
  constructor(maxSize = 200, ttlMs = 10 * 60 * 1000) {
    this.maxSize = maxSize;
    this.ttlMs = ttlMs;
    this.map = new Map();
  }

  get(key) {
    const entry = this.map.get(key);
    if (!entry) return null;
    if (Date.now() - entry.t > this.ttlMs) {
      this.map.delete(key);
      return null;
    }
    // Refresh recency
    this.map.delete(key);
    this.map.set(key, entry);
    return entry.v;
  }

  set(key, value) {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, { v: value, t: Date.now() });
    if (this.map.size > this.maxSize) {
      // Drop oldest
      const oldest = this.map.keys().next().value;
      this.map.delete(oldest);
    }
  }

  has(key) { return this.get(key) !== null; }

  clear() { this.map.clear(); }

  get stats() {
    return { size: this.map.size, maxSize: this.maxSize, ttlMs: this.ttlMs };
  }
}

module.exports = SimpleLRU;
