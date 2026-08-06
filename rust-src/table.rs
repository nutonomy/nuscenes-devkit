use std::collections::HashMap;
use std::sync::Arc;

use ahash::RandomState;
use rayon::prelude::*;

pub trait AsRefToken: Send + Sync {
    fn as_ref_token(&self) -> [u8; 16];
}

/// Represents a table of data in nuScenes
pub struct Table<T> {
    pub data: Arc<Box<[T]>>,
    pub index: HashMap<[u8; 16], usize, RandomState>,
}

impl<T: AsRefToken> Table<T> {
    pub fn new(data: Box<[T]>) -> Self {
        let index = Table::build_index(&data);
        let data = Arc::new(data);
        Table { data, index }
    }

    pub fn get(&self, token: &[u8; 16]) -> Option<&T> {
        self.index.get(token).map(|&idx| &self.data[idx])
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    fn build_index(data: &[T]) -> HashMap<[u8; 16], usize, RandomState> {
        data.par_iter().enumerate().map(|(i, item)| (item.as_ref_token(), i)).collect()
    }
}

impl<T: AsRefToken> From<Box<[T]>> for Table<T> {
    fn from(data: Box<[T]>) -> Self {
        Table::new(data)
    }
}

impl<'a, T> IntoIterator for &'a Table<T> {
    type IntoIter = std::slice::Iter<'a, T>;
    type Item = &'a T;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}
