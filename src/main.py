import os, random
import zipfile, json, tarfile
import supervisely_lib as sly
import glob
from supervisely_lib.io.fs import get_file_name, get_file_name_with_ext
from pathlib import Path
import shutil
import cv2
from collections import defaultdict
from supervisely_lib.video_annotation.video_tag import VideoTag
from supervisely_lib.video_annotation.video_tag_collection import VideoTagCollection
from PIL import Image
import numpy as np


my_app = sly.AppService()
TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
INPUT_DIR = os.environ.get("modal.state.slyFolder")
#INPUT_FILE = os.environ.get("modal.state.slyFile")
#PROJECT_NAME = 'DAVIS2017'
DATASET_NAME = 'ds'
EXTARACT_DIR_NAME = 'DAVIS'
logger = sly.logger
#archive_ext = '.zip'
frame_rate = 10
video_ext = '.mp4'
train_tag = 'train'
val_tag = 'val'
NECESSARY_ITEMS = ['JPEGImages', 'ImageSets', 'davis_semantics.json']
POSSIBLE_ITEMS = ['Annotations_unsupervised', 'Annotations']
POSSIBLE_SUBDIRS = ['480p', 'Full-Resolution']
SETS_YEAR = '2017'
images_ext = '.jpg'
anns_ext = '.png'
first_image_name = '00000.jpg'


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

    storage_dir = my_app.data_dir

    #if INPUT_DIR:
    cur_files_path = INPUT_DIR
    extract_dir = os.path.join(storage_dir, str(Path(cur_files_path).parent).lstrip("/"))
    input_dir = os.path.join(extract_dir, Path(cur_files_path).name)
    archive_path = os.path.join(storage_dir, cur_files_path.split("/")[-2] + ".tar")
    project_name = Path(cur_files_path).name
    # else:
    #     cur_files_path = INPUT_FILE
    #     extract_dir = os.path.join(storage_dir, get_file_name(cur_files_path))
    #     archive_path = os.path.join(storage_dir, get_file_name_with_ext(cur_files_path))
    #     project_name = get_file_name(INPUT_FILE)
    #     input_dir = extract_dir

    api.file.download(TEAM_ID, cur_files_path, archive_path)

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            archive.extractall(extract_dir)
    else:
        raise Exception("No such file".format(archive_path))

    for curr_arch_name in os.listdir(input_dir):
        curr_arch_path = os.path.join(input_dir, curr_arch_name)
        if zipfile.is_zipfile(curr_arch_path):
            with zipfile.ZipFile(curr_arch_path, 'r') as archive:
                archive.extractall(input_dir)
        else:
            logger.warn('Instance {} is not zip archive, it will be skipped'.format(curr_arch_name))
            continue

    working_dir = os.path.join(input_dir, EXTARACT_DIR_NAME)

    check_input_data(working_dir)

    if sly.fs.dir_exists(os.path.join(working_dir, POSSIBLE_ITEMS[0])):
        annotations_path = os.path.join(working_dir, POSSIBLE_ITEMS[0])
    elif sly.fs.dir_exists(os.path.join(working_dir, POSSIBLE_ITEMS[1])):
        annotations_path = os.path.join(working_dir, POSSIBLE_ITEMS[1])
    else:
        logger.warn(
            'There is no {} or {} folder in input data, but it must be.'.format(POSSIBLE_ITEMS[0], POSSIBLE_ITEMS[1]))
        my_app.stop()

    images_path = os.path.join(working_dir, NECESSARY_ITEMS[0])
    if sly.fs.dir_exists(os.path.join(images_path, POSSIBLE_SUBDIRS[0])):
        imgs_path = os.path.join(images_path, POSSIBLE_SUBDIRS[0])
    elif sly.fs.dir_exists(os.path.join(images_path, POSSIBLE_SUBDIRS[1])):
        imgs_path = os.path.join(images_path, POSSIBLE_SUBDIRS[1])
    else:
        logger.warn(
            'There is no {} or {} folder in input images data, but it must be.'.format(POSSIBLE_SUBDIRS[0],
                                                                                            POSSIBLE_SUBDIRS[1]))

    sets_path = os.path.join(working_dir, NECESSARY_ITEMS[1])
    semantics_path = os.path.join(working_dir, NECESSARY_ITEMS[2])
    semantics_json = sly.json.load_json_file(semantics_path)

    if sly.fs.dir_exists(os.path.join(annotations_path, POSSIBLE_SUBDIRS[0])):
        anns_dir = os.path.join(annotations_path, POSSIBLE_SUBDIRS[0])
    elif sly.fs.dir_exists(os.path.join(annotations_path, POSSIBLE_SUBDIRS[1])):
        anns_dir = os.path.join(annotations_path, POSSIBLE_SUBDIRS[1])
    else:
        logger.warn(
            'There is no {} or {} folder in input annotations data, but it must be.'.format(POSSIBLE_SUBDIRS[0],
                                                                                            POSSIBLE_SUBDIRS[1]))
        my_app.stop()

    if sly.fs.dir_exists(os.path.join(sets_path, SETS_YEAR)):
        train_val_path = os.path.join(sets_path, SETS_YEAR)
    else:
        logger.warn('There is no {} folder in input data, but it must be'.format(os.path.join(NECESSARY_ITEMS[1], SETS_YEAR)))
        my_app.stop()

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

    #=====================================================================================================
    new_project = api.project.create(WORKSPACE_ID, project_name, type=sly.ProjectType.VIDEOS,
                                     change_name_if_conflict=True)

    tag_meta_train = sly.TagMeta(train_tag, sly.TagValueType.NONE)
    tag_meta_val = sly.TagMeta(val_tag, sly.TagValueType.NONE)
    tag_collection = sly.TagMetaCollection([tag_meta_train, tag_meta_val])
    meta = sly.ProjectMeta(tag_metas=tag_collection)
    api.project.update_meta(new_project.id, meta.to_json())
    new_dataset = api.dataset.create(new_project.id, sly.fs.get_file_name(DATASET_NAME), change_name_if_conflict=True)
    # =====================================================================================================
    obj_classes = {}

    for imgs_dir in os.listdir(imgs_path):
        curr_imgs_path = os.path.join(imgs_path, imgs_dir)
        curr_anns_path = os.path.join(anns_dir, imgs_dir)
        if not sly.fs.dir_exists(curr_anns_path):
            logger.warn('There is no annotations for {} folder'.format(curr_imgs_path))
            continue

        video_objects = {}
        curr_semantic_classes = semantics_json[imgs_dir]
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
        video_path = os.path.join(extract_dir, imgs_dir + video_ext)
        img = cv2.imread(os.path.join(curr_imgs_path, first_image_name))
        img_size = (img.shape[1], img.shape[0])
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MP4V'), frame_rate, img_size)

        frames = []
        for idx in range(len(images)):
            image_name = str(idx).zfill(5) + images_ext
            curr_im_path = os.path.join(curr_imgs_path, image_name)
            if not sly.fs.file_exists(curr_im_path):
                logger.warn('There is no image with name {}, but it must be. Folder {} will be skip.'.format(image_name, imgs_dir))
                break

            curr_im = cv2.imread(curr_im_path)
            if (curr_im.shape[1], curr_im.shape[0]) != img_size:
                logger.warn(
                    'Image {} shape not correspond to {} image shape in {} folder, this folder will be skip.'.format(
                        image_name, first_image_name, imgs_dir))

            ann_name = str(idx).zfill(5) + anns_ext
            curr_ann_path = os.path.join(curr_anns_path, ann_name)
            if not sly.fs.file_exists(curr_im_path):
                logger.warn('There is no annotation with name {}, but it must be. Folder {} will be skip.'.format(ann_name, imgs_dir))
                break

            curr_ann = Image.open(curr_ann_path)
            ann_objects = curr_ann.getcolors()
            mask_all = np.asarray(curr_ann)
            figures = []
            for ann_obj_idx in range(1, len(ann_objects)):
                obj_id = ann_objects[ann_obj_idx][1]
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

    my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": TEAM_ID,
        "WORKSPACE_ID": WORKSPACE_ID
    })
    my_app.run(initial_events=[{"command": "import_davis"}])


if __name__ == '__main__':
    sly.main_wrapper("main", main)

