/// 257-bucket priority queue (one per gray level 0..=255, plus sentinel at 256).
/// Each bucket is a LIFO stack of `usize` indices.
///
/// Mirrors the C++ heap where `heap_start[i]` points to a stack of pixel indices
/// for gray level `i`, with `heap_start[i][0]` as a count/sentinel.
pub struct BucketHeap {
    storage: Vec<usize>,
    /// For each bucket, the base offset in `storage` where its elements start.
    bucket_starts: [usize; 257],
    /// For each bucket, the current stack top offset (relative to `storage`).
    /// Points to the last pushed element. When equal to `bucket_starts[i]`,
    /// the bucket is empty (the slot at bucket_starts[i] is the sentinel).
    cursors: [usize; 257],
}

impl BucketHeap {
    /// Create a new BucketHeap from level_size histogram.
    /// `level_sizes[i]` = number of pixels at gray level `i`.
    pub fn new(level_sizes: &[u32; 257]) -> Self {
        let total: usize = level_sizes.iter().map(|&s| s as usize + 1).sum();
        let mut storage = vec![0usize; total];
        let mut bucket_starts = [0usize; 257];
        let mut cursors = [0usize; 257];

        let mut offset = 0;
        for i in 0..257 {
            bucket_starts[i] = offset;
            cursors[i] = offset; // empty: cursor == start (sentinel position)
            storage[offset] = 0; // sentinel
            offset += level_sizes[i] as usize + 1;
        }

        Self {
            storage,
            bucket_starts,
            cursors,
        }
    }

    pub fn push(&mut self, bucket: usize, index: usize) {
        self.cursors[bucket] += 1;
        let pos = self.cursors[bucket];
        self.storage[pos] = index;
    }

    pub fn pop(&mut self, bucket: usize) -> Option<usize> {
        if self.cursors[bucket] == self.bucket_starts[bucket] {
            return None;
        }
        let pos = self.cursors[bucket];
        let val = self.storage[pos];
        self.cursors[bucket] -= 1;
        Some(val)
    }

    pub fn is_empty(&self, bucket: usize) -> bool {
        self.cursors[bucket] == self.bucket_starts[bucket]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heap_from_level_sizes() {
        let mut sizes = [0u32; 257];
        sizes[0] = 3;
        sizes[1] = 2;
        sizes[2] = 0;
        let heap = BucketHeap::new(&sizes);
        // bucket 0: 1 sentinel + 3 slots = 4 elements, starts at 0
        assert_eq!(heap.bucket_starts[0], 0);
        // bucket 1: starts at 4
        assert_eq!(heap.bucket_starts[1], 4);
        // bucket 2: starts at 4 + 3 = 7
        assert_eq!(heap.bucket_starts[2], 7);
    }

    #[test]
    fn test_push_pop_single_bucket() {
        let mut sizes = [0u32; 257];
        sizes[42] = 5;
        let mut heap = BucketHeap::new(&sizes);

        heap.push(42, 100);
        heap.push(42, 200);
        heap.push(42, 300);

        // LIFO order
        assert_eq!(heap.pop(42), Some(300));
        assert_eq!(heap.pop(42), Some(200));
        assert_eq!(heap.pop(42), Some(100));
        assert_eq!(heap.pop(42), None);
    }

    #[test]
    fn test_push_pop_multiple_buckets() {
        let mut sizes = [0u32; 257];
        sizes[0] = 2;
        sizes[1] = 2;
        let mut heap = BucketHeap::new(&sizes);

        heap.push(0, 10);
        heap.push(1, 20);
        heap.push(0, 30);

        assert_eq!(heap.pop(0), Some(30));
        assert_eq!(heap.pop(1), Some(20));
        assert_eq!(heap.pop(0), Some(10));
        assert!(heap.is_empty(0));
        assert!(heap.is_empty(1));
    }

    #[test]
    fn test_empty_bucket() {
        let sizes = [0u32; 257];
        let mut heap = BucketHeap::new(&sizes);
        assert!(heap.is_empty(0));
        assert_eq!(heap.pop(0), None);
    }
}
