from multiprocessing.pool import ThreadPool
from PIL import Image
from pathlib import Path


class ImageSaver:
    # frugal attempt to make IO non-blocking, thus speeding up the simulation
    _thread_pool: ThreadPool = None

    @staticmethod
    def _save_img_helper(
        img, folder: str, name: str, last_ptr_name: str, in_folder: bool
    ):
        folder = Path(folder)
        Image.fromarray(img).save(folder / name)
        last_ptr_name = folder / last_ptr_name if in_folder else Path(last_ptr_name)
        last_ptr_name.unlink(missing_ok=True)
        last_ptr_name.symlink_to(name if in_folder else folder / name)

    @staticmethod
    def _save_img(img, folder: str, path: str, last_ptr_path: str, in_folder: bool):
        if ImageSaver._thread_pool is None:
            ImageSaver._thread_pool = ThreadPool(3)
        # ImageSaver._save_img_helper(img, folder, path, last_ptr_path, in_folder)
        ImageSaver._thread_pool.apply_async(
            ImageSaver._save_img_helper, (img, folder, path, last_ptr_path, in_folder)
        )

    @staticmethod
    def save_rgb(rgb, t, folder="imgs", last_ptr_name="_last.png"):
        name = f"{str(int(t)).zfill(5)}.png"
        ImageSaver._save_img(rgb, folder, name, last_ptr_name, False)

    @staticmethod
    def save_flow_img(flow_img, cnt, folder="imgs", last_ptr_name="_flow_last.png"):
        name = f"flow_{str(cnt).zfill(5)}.png"
        ImageSaver._save_img(flow_img, folder, name, last_ptr_name, True)
