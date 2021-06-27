import os
import zipfile
import supervisely_lib as sly
from supervisely_lib.io.fs import get_file_name, get_file_name_with_ext
import cv2
from collections import defaultdict
from supervisely_lib.video_annotation.video_tag import VideoTag
from supervisely_lib.video_annotation.video_tag_collection import VideoTagCollection
from PIL import Image
import numpy as np
from supervisely_lib.io.fs import download
from functools import partial
import requests


my_app = sly.AppService()
TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
DATASET_NAME = 'ds'
EXTARACT_DIR_NAME = 'DAVIS'
logger = sly.logger
frame_rate = 10
video_ext = '.mp4'
train_tag = 'train'
val_tag = 'val'
test_tag = 'test'
NECESSARY_ITEMS = ['JPEGImages', 'ImageSets', 'davis_semantics.json']
images_ext = '.jpg'
anns_ext = '.png'
first_image_name = '00000.jpg'
project_name = 'davis2017'
work_dir = 'davis_data'
trainval = 'trainval'


train_val_480_url = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-480p.zip'
train_val_full_url = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-Full-Resolution.zip'
anns_480_url = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017_semantics-480p.zip'
anns_full_url = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017_semantics-Full-resolution.zip'
test_dev_480_url = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2019-Unsupervised-test-dev-480p.zip'
test_dev_full_url = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2019-Unsupervised-test-dev-Full-Resolution.zip'
test_chall_480_url = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2019-Unsupervised-test-challenge-480p.zip'
test_chall_full_url = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2019-Unsupervised-test-challenge-Full-Resolution.zip'

test_dev_480_url_2017 = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip'
test_dev_full_url_2017 = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-Full-Resolution.zip'
test_chall_480_url_2017 = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-challenge-480p.zip'
test_chall_full_url_2017 = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-challenge-Full-Resolution.zip'

train_val_480_url_semi = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip'
train_val_full_url_semi = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip'

train_val_2016_url = 'https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip'

resolution = os.environ['modal.state.resolution']
datasets = os.environ['modal.state.currDataset']
davis_year = os.environ['modal.state.Davis']
davis_type = os.environ['modal.state.type']

logger.warn('resolution {}'.format(resolution))
logger.warn('datasets {}'.format(datasets))
logger.warn('davis_year {}'.format(davis_year))
logger.warn('davis_type {}'.format(davis_type))

if len(datasets) != 2:
    datasets = os.environ['modal.state.currDataset']
    datasets = datasets[1:-1].replace('\'', '')
    datasets = datasets.replace(' ', '').split(',')
else:
    logger.warn('You have no choose any dataset')
    my_app.stop()

LINKS = []
if davis_year == '2017':
    if resolution == '480p':
        for ds in datasets:
            if ds == 'TrainVal' and davis_type == 'Unsupervised':
                LINKS.extend([train_val_480_url, anns_480_url])
            if ds == 'TrainVal' and davis_type == 'Semi-supervised':
                LINKS.extend([train_val_480_url_semi, anns_480_url])
            if ds == 'TestDev' and davis_type == 'Unsupervised':
                LINKS.append(test_dev_480_url)
            if ds == 'TestChallenge' and davis_type == 'Unsupervised':
                LINKS.append(test_chall_480_url)
            if ds == 'TestDev' and davis_type == 'Semi-supervised':
                LINKS.append(test_dev_480_url_2017)
            if ds == 'TestChallenge' and davis_type == 'Semi-supervised':
                LINKS.append(test_chall_480_url_2017)
    else:
        for ds in datasets:
            if ds == 'TrainVal' and davis_type == 'Unsupervised':
                LINKS.extend([train_val_full_url, anns_full_url])
            if ds == 'TrainVal' and davis_type == 'Semi-supervised':
                LINKS.extend([train_val_full_url_semi, anns_full_url])
            if ds == 'TestDev' and davis_type == 'Unsupervised':
                LINKS.append(test_dev_full_url)
            if ds == 'TestChallenge' and davis_type == 'Unsupervised':
                LINKS.append(test_chall_full_url)
            if ds == 'TestDev' and davis_type == 'Semi-supervised':
                LINKS.append(test_dev_full_url_2017)
            if ds == 'TestChallenge' and davis_type == 'Semi-supervised':
                LINKS.append(test_chall_full_url_2017)
else:
    pass


def check_input_data(working_dir):
    all_items = os.listdir(working_dir)
    necessary_itm = set(NECESSARY_ITEMS) - set(all_items)
    if len(necessary_itm) != 0:
        logger.warn('There is no {} items in input data, but it must be'.format(necessary_itm))
        my_app.stop()


def check_imgs_to_anns(img_paths, ann_paths, data_folder):
    imgs = os.listdir(img_paths)
    anns = os.listdir(ann_paths)
    if len(imgs) != len(anns):
        logger.warn('Number of images not equal to number of annotations in {} folder'.format(data_folder))
        return None
    return True


@my_app.callback("import_davis")
@sly.timeit
def import_davis(api: sly.Api, task_id, context, state, app_logger):

    def update_progress(count, index, api: sly.Api, task_id, progress: sly.Progress):
        progress.iters_done(count)

    def get_progress_cb(index, message, total, is_size=False, min_report_percent=5, upd_func=update_progress):
        progress = sly.Progress(message, total, is_size=is_size, min_report_percent=min_report_percent)
        progress_cb = partial(upd_func, index=index, api=api, task_id=task_id, progress=progress)
        progress_cb(0)
        return progress_cb

    storage_dir = my_app.data_dir
    work_dir_path = os.path.join(storage_dir, work_dir)
    sly.io.fs.mkdir(work_dir_path)

    for curr_url in LINKS:
        arch_name = get_file_name_with_ext(curr_url)
        archive_path = os.path.join(work_dir_path, arch_name)
        response = requests.head(curr_url, allow_redirects=True)
        sizeb = int(response.headers.get('content-length', 0))
        progress_cb = get_progress_cb(6, "Download {}".format(arch_name), sizeb, is_size=True, min_report_percent=1)
        download(curr_url, archive_path, my_app.cache, progress_cb)

    dirs_for_prepare = []
    for curr_arch_name in os.listdir(work_dir_path):
        curr_arch_path = os.path.join(work_dir_path, curr_arch_name)
        if sly.io.fs.file_exists(curr_arch_path):
            if 'DAVIS-2017-Unsupervised' in curr_arch_path or 'DAVIS-2017_semantics' in curr_arch_path:
                subdir = trainval
            elif 'dev' in curr_arch_path:
                subdir = 'test_dev'
            elif 'challenge' in curr_arch_path:
                subdir = 'test_challenge'
            else:
                logger.warn('Unknown archive name {}'.format(curr_arch_name))
                my_app.stop()

            curr_extract_dir = os.path.join(work_dir_path, subdir)
            dirs_for_prepare.append(os.path.join(curr_extract_dir, EXTARACT_DIR_NAME))

        if zipfile.is_zipfile(curr_arch_path):
            with zipfile.ZipFile(curr_arch_path, 'r') as archive:
                archive.extractall(curr_extract_dir)
        else:
            logger.warn('Archive cannot be unpacked {}'.format(curr_arch_name))
            my_app.stop()

    dirs_for_prepare = set(dirs_for_prepare)
    # =====================================================================================================
    new_project = api.project.create(WORKSPACE_ID, project_name, type=sly.ProjectType.VIDEOS,
                                     change_name_if_conflict=True)

    meta = sly.ProjectMeta()
    new_dataset = api.dataset.create(new_project.id, DATASET_NAME, change_name_if_conflict=True)
    # =====================================================================================================
    for working_dir in dirs_for_prepare:
        if trainval in working_dir:
            check_input_data(working_dir)
            anns_dir = os.path.join(working_dir, 'Annotations_unsupervised', resolution)
            imgs_path = os.path.join(working_dir, 'JPEGImages', resolution)
            train_val_path = os.path.join(working_dir, 'ImageSets/2017')
            semantics_path = os.path.join(working_dir, 'davis_semantics.json')
            semantics_json = sly.json.load_json_file(semantics_path)

            tag_meta_train = sly.TagMeta(train_tag, sly.TagValueType.NONE)
            tag_meta_val = sly.TagMeta(val_tag, sly.TagValueType.NONE)
            tag_collection = sly.TagMetaCollection([tag_meta_train, tag_meta_val])
            train_val_meta = sly.ProjectMeta(tag_metas=tag_collection)
            meta = meta.merge(train_val_meta)
            api.project.update_meta(new_project.id, meta.to_json())

            train_val_names = defaultdict(list)

            for curr_file in os.listdir(train_val_path):
                curr_file_path = os.path.join(train_val_path, curr_file)
                curr_file_name = sly.fs.get_file_name(curr_file_path)
                if curr_file_name not in [train_tag, val_tag]:
                    logger.warn('File {} not train.txt or val.txt, it will be skip.'.format(curr_file_name))
                    continue
                with open(curr_file_path, "r") as file:
                    all_lines = file.readlines()
                    for line in all_lines:
                        line = line.split('\n')[0].split(' ')
                        train_val_names[curr_file_name].append(line[0])

            obj_classes = {}
            for imgs_dir in os.listdir(imgs_path):
                curr_imgs_path = os.path.join(imgs_path, imgs_dir)
                curr_anns_path = os.path.join(anns_dir, imgs_dir)
                if not sly.fs.dir_exists(curr_anns_path):
                    logger.warn('There is no annotations for {} folder'.format(curr_imgs_path))
                    continue

                video_objects = {}
                curr_semantic_classes = semantics_json[imgs_dir]
                if imgs_dir == 'color-run' and davis_type == 'Unsupervised': # davis2017 Unsupervised bug in 'color-run'
                    curr_semantic_classes['4'] = 'person'
                for id, obj_name in curr_semantic_classes.items():
                    if obj_name not in obj_classes.keys():
                        obj_class = sly.ObjClass(obj_name, sly.Bitmap)
                        obj_classes[obj_name] = obj_class
                    video_objects[int(id)] = sly.VideoObject(obj_classes[obj_name])

                new_meta = sly.ProjectMeta(sly.ObjClassCollection(list(obj_classes.values())))
                meta = meta.merge(new_meta)
                api.project.update_meta(new_project.id, meta.to_json())
                if not check_imgs_to_anns(curr_imgs_path, curr_anns_path, imgs_dir):
                    continue

                images = os.listdir(curr_imgs_path)
                progress = sly.Progress('Create video {}'.format(imgs_dir), len(images), app_logger)
                video_path = os.path.join(work_dir_path, imgs_dir + video_ext)
                img = cv2.imread(os.path.join(curr_imgs_path, first_image_name))
                img_size = (img.shape[1], img.shape[0])
                video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MP4V'), frame_rate, img_size)

                frames = []
                really_video_classes_ids = []
                for idx in range(len(images)):
                    image_name = str(idx).zfill(5) + images_ext
                    curr_im_path = os.path.join(curr_imgs_path, image_name)
                    if not sly.fs.file_exists(curr_im_path):
                        logger.warn(
                            'There is no image with name {}, but it must be. Folder {} will be skip.'.format(image_name,
                                                                                                             imgs_dir))
                        break

                    curr_im = cv2.imread(curr_im_path)
                    if (curr_im.shape[1], curr_im.shape[0]) != img_size:
                        logger.warn(
                            'Image {} shape not correspond to {} image shape in {} folder, this folder will be skip.'.format(
                                image_name, first_image_name, imgs_dir))

                    ann_name = str(idx).zfill(5) + anns_ext
                    curr_ann_path = os.path.join(curr_anns_path, ann_name)
                    if not sly.fs.file_exists(curr_im_path):
                        logger.warn(
                            'There is no annotation with name {}, but it must be. Folder {} will be skip.'.format(
                                ann_name, imgs_dir))
                        break

                    curr_ann = Image.open(curr_ann_path)
                    ann_objects = curr_ann.getcolors()
                    mask_all = np.asarray(curr_ann)
                    figures = []
                    for ann_obj_idx in range(1, len(ann_objects)):
                        obj_id = ann_objects[ann_obj_idx][1]
                        if obj_id not in really_video_classes_ids:
                            really_video_classes_ids.append(obj_id)
                        mask = mask_all == obj_id
                        if len(np.unique(mask)) == 1:
                            continue
                        if img_size[1] % 2 == 1:
                            mask[mask.shape[0] - 1, :] = False
                        if img_size[0] % 2 == 1:
                            mask[:, mask.shape[1] - 1] = False
                        geom = sly.Bitmap(mask)
                        if not video_objects.get(obj_id):
                            continue
                        curr_video_obj = video_objects[obj_id]
                        figure = sly.VideoFigure(curr_video_obj, geom, idx)
                        figures.append(figure)
                    new_frame = sly.Frame(idx, figures)
                    frames.append(new_frame)

                    video.write(curr_im)
                    progress.iter_done_report()
                #========================================================================================
                keys_to_del = []
                for key in video_objects.keys():
                    if int(key) not in really_video_classes_ids:
                        keys_to_del.append(key)
                for k in keys_to_del:
                    video_objects.pop(k)
                # ========================================================================================
                video.release()

                file_info = api.video.upload_paths(new_dataset.id, [imgs_dir], [video_path])
                new_frames_collection = sly.FrameCollection(frames)
                new_objects = sly.VideoObjectCollection(list(video_objects.values()))
                if imgs_dir in train_val_names['train']:
                    tag = VideoTag(tag_meta_train)
                elif imgs_dir in train_val_names['val']:
                    tag = VideoTag(tag_meta_val)
                else:
                    logger.warn(
                        'There is no folder name {} in train.txt or val.txt. The video will be assigned a train tag value'.format(
                            imgs_dir))
                    tag = VideoTag(tag_meta_train)

                tag_collection = VideoTagCollection([tag])
                ann = sly.VideoAnnotation((img_size[1], img_size[0]), len(frames), objects=new_objects,
                                          frames=new_frames_collection, tags=tag_collection)
                logger.info('Create annotation for video {}'.format(imgs_dir))
                api.video.annotation.append(file_info[0].id, ann)

        else:
            imgs_path = os.path.join(working_dir, 'JPEGImages', resolution)
            tag_meta_test = sly.TagMeta(test_tag, sly.TagValueType.NONE)
            tag_collection = sly.TagMetaCollection([tag_meta_test])
            test_meta = sly.ProjectMeta(tag_metas=tag_collection)
            meta = meta.merge(test_meta)
            api.project.update_meta(new_project.id, meta.to_json())

            for imgs_dir in os.listdir(imgs_path):
                curr_imgs_path = os.path.join(imgs_path, imgs_dir)
                images = os.listdir(curr_imgs_path)
                progress = sly.Progress('Create video {}'.format(imgs_dir), len(images), app_logger)
                video_path = os.path.join(work_dir_path, imgs_dir + video_ext)
                img = cv2.imread(os.path.join(curr_imgs_path, first_image_name))
                img_size = (img.shape[1], img.shape[0])
                video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MP4V'), frame_rate, img_size)
                for idx in range(len(images)):
                    image_name = str(idx).zfill(5) + images_ext
                    curr_im_path = os.path.join(curr_imgs_path, image_name)
                    if not sly.fs.file_exists(curr_im_path):
                        logger.warn(
                            'There is no image with name {}, but it must be. Folder {} will be skip.'.format(image_name,
                                                                                                             imgs_dir))
                        break

                    curr_im = cv2.imread(curr_im_path)
                    if (curr_im.shape[1], curr_im.shape[0]) != img_size:
                        logger.warn(
                            'Image {} shape not correspond to {} image shape in {} folder, this folder will be skip.'.format(
                                image_name, first_image_name, imgs_dir))

                    video.write(curr_im)
                    progress.iter_done_report()
                video.release()

                file_info = api.video.upload_paths(new_dataset.id, [imgs_dir], [video_path])
                tag = VideoTag(tag_meta_test)
                tag_collection = VideoTagCollection([tag])
                ann = sly.VideoAnnotation((img_size[1], img_size[0]), len(images), tags=tag_collection)
                logger.info('Create annotation for video {}'.format(imgs_dir))
                api.video.annotation.append(file_info[0].id, ann)

    my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": TEAM_ID,
        "WORKSPACE_ID": WORKSPACE_ID
    })
    my_app.run(initial_events=[{"command": "import_davis"}])


if __name__ == '__main__':
    sly.main_wrapper("main", main)

