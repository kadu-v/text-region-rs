pub struct BlockMemory<T> {
    blocks: Vec<Vec<T>>,
    block_size: usize,
    block_mask: usize,
    block_shift: u32,
    pub element_count: usize,
}

impl<T> BlockMemory<T> {
    pub fn new(block_size_log2: u32) -> Self {
        let block_size = 1usize << block_size_log2;
        Self {
            blocks: Vec::new(),
            block_size,
            block_mask: block_size - 1,
            block_shift: block_size_log2,
            element_count: 0,
        }
    }

    pub fn init(&mut self, block_size_log2: u32) {
        self.block_size = 1usize << block_size_log2;
        self.block_mask = self.block_size - 1;
        self.block_shift = block_size_log2;
        self.element_count = 0;
        self.blocks.clear();
    }

    pub fn add(&mut self, item: T) -> usize {
        let index = self.element_count;
        let block_idx = index >> self.block_shift;
        if block_idx >= self.blocks.len() {
            self.blocks.push(Vec::with_capacity(self.block_size));
        }
        self.blocks[block_idx].push(item);
        self.element_count += 1;
        index
    }

    pub fn get(&self, index: usize) -> &T {
        let block_idx = index >> self.block_shift;
        let within_idx = index & self.block_mask;
        &self.blocks[block_idx][within_idx]
    }

    pub fn get_mut(&mut self, index: usize) -> &mut T {
        let block_idx = index >> self.block_shift;
        let within_idx = index & self.block_mask;
        &mut self.blocks[block_idx][within_idx]
    }

    pub fn len(&self) -> usize {
        self.element_count
    }

    pub fn is_empty(&self) -> bool {
        self.element_count == 0
    }

    pub fn iter(&self) -> BlockMemoryIter<'_, T> {
        BlockMemoryIter {
            memory: self,
            index: 0,
        }
    }

    pub fn iter_mut(&mut self) -> BlockMemoryIterMut<'_, T> {
        BlockMemoryIterMut {
            blocks: &mut self.blocks,
            block_shift: self.block_shift,
            block_mask: self.block_mask,
            element_count: self.element_count,
            index: 0,
        }
    }
}

pub struct BlockMemoryIter<'a, T> {
    memory: &'a BlockMemory<T>,
    index: usize,
}

impl<'a, T> Iterator for BlockMemoryIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.memory.element_count {
            return None;
        }
        let item = self.memory.get(self.index);
        self.index += 1;
        Some(item)
    }
}

pub struct BlockMemoryIterMut<'a, T> {
    blocks: &'a mut Vec<Vec<T>>,
    block_shift: u32,
    block_mask: usize,
    element_count: usize,
    index: usize,
}

impl<'a, T> Iterator for BlockMemoryIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.element_count {
            return None;
        }
        let block_idx = self.index >> self.block_shift;
        let within_idx = self.index & self.block_mask;
        self.index += 1;
        // SAFETY: Each call yields a unique element, so no aliasing occurs
        let ptr = &mut self.blocks[block_idx][within_idx] as *mut T;
        Some(unsafe { &mut *ptr })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_block_memory() {
        let bm = BlockMemory::<i32>::new(2);
        assert_eq!(bm.len(), 0);
        assert!(bm.is_empty());
    }

    #[test]
    fn test_add_single_element() {
        let mut bm = BlockMemory::<i32>::new(2);
        let idx = bm.add(42);
        assert_eq!(idx, 0);
        assert_eq!(*bm.get(0), 42);
        assert_eq!(bm.len(), 1);
    }

    #[test]
    fn test_add_across_block_boundary() {
        let mut bm = BlockMemory::<i32>::new(2); // block_size = 4
        for i in 0..5 {
            bm.add(i * 10);
        }
        assert_eq!(bm.len(), 5);
        for i in 0..5 {
            assert_eq!(*bm.get(i), i as i32 * 10);
        }
    }

    #[test]
    fn test_add_many_elements() {
        let mut bm = BlockMemory::<i32>::new(4); // block_size = 16
        for i in 0..1000 {
            bm.add(i);
        }
        assert_eq!(bm.len(), 1000);
        for i in 0..1000 {
            assert_eq!(*bm.get(i), i as i32);
        }
    }

    #[test]
    fn test_get_mut() {
        let mut bm = BlockMemory::<i32>::new(2);
        bm.add(10);
        bm.add(20);
        *bm.get_mut(0) = 99;
        assert_eq!(*bm.get(0), 99);
        assert_eq!(*bm.get(1), 20);
    }

    #[test]
    fn test_iter() {
        let mut bm = BlockMemory::<i32>::new(2); // block_size = 4
        for i in 0..7 {
            bm.add(i);
        }
        let collected: Vec<i32> = bm.iter().copied().collect();
        assert_eq!(collected, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_init_reset() {
        let mut bm = BlockMemory::<i32>::new(2);
        for i in 0..10 {
            bm.add(i);
        }
        assert_eq!(bm.len(), 10);

        bm.init(3); // reset with new block_size = 8
        assert_eq!(bm.len(), 0);
        assert!(bm.is_empty());

        bm.add(100);
        assert_eq!(bm.len(), 1);
        assert_eq!(*bm.get(0), 100);
    }
}
