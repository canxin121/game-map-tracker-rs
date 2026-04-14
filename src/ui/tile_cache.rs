use std::{collections::HashMap, sync::Arc};

use futures_util::FutureExt;
use gpui::{
    App, AppContext, Asset as _, AssetLogger, Entity, ImageAssetLoader, ImageCacheError,
    ImageCacheItem, RenderImage, Resource, Window, hash,
};

#[derive(Debug)]
pub struct TileImageCache {
    max_items: usize,
    max_bytes: usize,
    bytes_used: usize,
    usages: Vec<u64>,
    cache: HashMap<u64, TileCacheEntry>,
}

#[derive(Debug)]
struct TileCacheEntry {
    item: ImageCacheItem,
    bytes: usize,
}

impl TileImageCache {
    pub fn new(max_items: usize, max_bytes: usize, cx: &mut App) -> Entity<Self> {
        let entity = cx.new(|_cx| Self {
            max_items: max_items.max(1),
            max_bytes: max_bytes.max(1),
            bytes_used: 0,
            usages: Vec::with_capacity(max_items.max(1)),
            cache: HashMap::with_capacity(max_items.max(1)),
        });

        cx.observe_release(&entity, |cache, cx| {
            for (_, mut entry) in std::mem::take(&mut cache.cache) {
                if let Some(Ok(image)) = entry.item.get() {
                    cx.drop_image(image, None);
                }
            }
            cache.usages.clear();
            cache.bytes_used = 0;
        })
        .detach();

        entity
    }

    pub fn load(
        &mut self,
        resource: &Resource,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Result<Arc<RenderImage>, ImageCacheError>> {
        let key = hash(resource);
        if self.cache.contains_key(&key) {
            self.touch(key);
            let result = {
                let entry = self
                    .cache
                    .get_mut(&key)
                    .expect("tile cache entry should exist after contains_key");
                entry.item.get()
            };
            self.refresh_entry_bytes(key);
            self.trim(window, cx);
            return result;
        }

        let future = AssetLogger::<ImageAssetLoader>::load(resource.clone(), cx);
        let task = cx.background_executor().spawn(future).shared();
        self.cache.insert(
            key,
            TileCacheEntry {
                item: ImageCacheItem::Loading(task.clone()),
                bytes: 0,
            },
        );
        self.usages.insert(0, key);
        self.trim(window, cx);

        let entity = window.current_view();
        window
            .spawn(cx, {
                async move |cx| {
                    _ = task.await;
                    cx.on_next_frame(move |_, cx| {
                        cx.notify(entity);
                    });
                }
            })
            .detach();

        None
    }

    fn touch(&mut self, key: u64) {
        if let Some(index) = self.usages.iter().position(|item| *item == key) {
            self.usages.remove(index);
        }
        self.usages.insert(0, key);
    }

    fn refresh_entry_bytes(&mut self, key: u64) {
        let Some(entry) = self.cache.get_mut(&key) else {
            return;
        };
        let Some(Ok(image)) = entry.item.get() else {
            return;
        };

        let next_bytes = render_image_bytes(&image);
        if next_bytes == entry.bytes {
            return;
        }

        self.bytes_used = self
            .bytes_used
            .saturating_sub(entry.bytes)
            .saturating_add(next_bytes);
        entry.bytes = next_bytes;
    }

    fn trim(&mut self, window: &mut Window, cx: &mut App) {
        while self.cache.len() > self.max_items
            || (self.bytes_used > self.max_bytes && self.cache.len() > 1)
        {
            let Some(oldest_key) = self.usages.pop() else {
                break;
            };
            let Some(mut entry) = self.cache.remove(&oldest_key) else {
                continue;
            };
            self.bytes_used = self.bytes_used.saturating_sub(entry.bytes);
            if let Some(Ok(image)) = entry.item.get() {
                cx.drop_image(image, Some(window));
            }
        }
    }
}

fn render_image_bytes(image: &RenderImage) -> usize {
    (0..image.frame_count())
        .filter_map(|frame_index| image.as_bytes(frame_index))
        .map(|bytes| bytes.len())
        .sum()
}
