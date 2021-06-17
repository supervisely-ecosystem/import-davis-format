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
INPUT_FILE = os.environ.get("modal.state.slyFile")
PROJECT_NAME = 'DAVIS2017'
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

# test = Image.open('/home/andrew/alex_work/app_data/data/davis_data/00000.png')
# colors = test.getcolors()
# palette = test.palette
# a = np.asarray(test)
# rgb = test.convert('RGB').getcolors()
# a=0

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

    if INPUT_DIR:
        cur_files_path = INPUT_DIR
        extract_dir = os.path.join(storage_dir, str(Path(cur_files_path).parent).lstrip("/"))
        input_dir = os.path.join(extract_dir, Path(cur_files_path).name)
        archive_path = os.path.join(storage_dir, cur_files_path.split("/")[-2] + ".tar")
        project_name = Path(cur_files_path).name
    else:
        cur_files_path = INPUT_FILE
        extract_dir = os.path.join(storage_dir, get_file_name(cur_files_path))
        archive_path = os.path.join(storage_dir, get_file_name_with_ext(cur_files_path))
        project_name = get_file_name(INPUT_FILE)
        input_dir = extract_dir

    # api.file.download(TEAM_ID, cur_files_path, archive_path)
    #
    # if tarfile.is_tarfile(archive_path):
    #     with tarfile.open(archive_path) as archive:
    #         archive.extractall(extract_dir)
    # else:
    #     raise Exception("No such file".format(INPUT_FILE))


    # for curr_arch_name in os.listdir(input_dir):
    #     curr_arch_path = os.path.join(input_dir, curr_arch_name)
    #     if zipfile.is_zipfile(curr_arch_path):
    #         with zipfile.ZipFile(curr_arch_path, 'r') as archive:
    #             archive.extractall(input_dir)
    #     else:
    #         logger.warn('Instance {} is not zip archive, it will be skipped'.format(curr_arch_name))  #TODO test it!!!
    #         continue


    working_dir = os.path.join(input_dir, EXTARACT_DIR_NAME)

    check_input_data(working_dir)  #TODO test it!!!

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
    elif sly.fs.dir_exists(os.path.join(annotations_path, POSSIBLE_SUBDIRS[0])):
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
            obj_class = sly.ObjClass(obj_name, sly.Bitmap)
            if obj_name not in obj_classes.keys():
                obj_classes[obj_name] = obj_class
            video_objects[id] = sly.VideoObject(obj_class)

        if not check_imgs_to_anns(curr_imgs_path, curr_anns_path, imgs_dir):
            continue

        images = os.listdir(curr_imgs_path)
        progress = sly.Progress('Create video', len(images), app_logger)
        video_path = os.path.join(extract_dir, imgs_dir + video_ext)
        img = cv2.imread(os.path.join(curr_imgs_path, first_image_name))
        img_size = (img.shape[1], img.shape[0])
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MP4V'), frame_rate, img_size)

        for idx in range(len(images)):
            image_name = str(idx).zfill(5) + images_ext
            curr_im_path = os.path.join(curr_imgs_path, image_name)
            if not sly.fs.file_exists(curr_im_path):
                logger.warn('There is no image with name {}, but it must be. Folder {} will be skip.'.format(image_name, imgs_dir)) #TODO make all imgs check in check_imgs_to_anns
                break

            curr_im = cv2.imread(curr_im_path)
            if (curr_im.shape[1], curr_im.shape[0]) != img_size:
                logger.warn(
                    'Image {} shape not correspond to {} image shape in {} folder, this folder will be skip.'.format(
                        image_name, first_image_name, imgs_dir))

            ann_name = str(idx).zfill(5) + anns_ext
            curr_ann_path = os.path.join(curr_anns_path, ann_name)
            if not sly.fs.file_exists(curr_im_path):
                logger.warn('There is no annotation with name {}, but it must be. Folder {} will be skip.'.format(ann_name, imgs_dir)) #TODO make all imgs check in check_imgs_to_anns
                break

            curr_ann = Image.open(curr_ann_path)
            ann_objects = curr_ann.getcolors()
            mask_all = np.asarray(curr_ann)
            for ann_obj_idx in range(1, len(ann_objects)):
                obj_id = ann_objects[ann_obj_idx][1]
                mask = 
                figure = sly.VideoFigure(video_objects[obj_id], mask, idx)




            a=0



            video.write(curr_im)
        progress.iter_done_report()
        video.release()

        file_info = api.video.upload_paths(new_dataset.id, [imgs_dir], [video_path])



        a = 0





    search_anns = os.path.join(input_dir, "annotations_*.json")
    anns_fine_paths = glob.glob(search_anns)
    if len(anns_fine_paths) == 0:
        logger.warn('There is no annotations in input data. Check your input format')
    anns_fine_paths.sort()

    ovis_classes = {}
    id_to_obj_classes = {}

    new_project = api.project.create(WORKSPACE_ID, project_name, type=sly.ProjectType.VIDEOS,
                                     change_name_if_conflict=True)

    tag_meta_train = sly.TagMeta(train_tag, sly.TagValueType.NONE)
    tag_meta_val = sly.TagMeta(val_tag, sly.TagValueType.NONE)
    tag_collection = sly.TagMetaCollection([tag_meta_train, tag_meta_val])
    meta = sly.ProjectMeta(tag_metas=tag_collection)
    api.project.update_meta(new_project.id, meta.to_json())

    for ann_path in anns_fine_paths:
        ann_name = str(Path(ann_path).name)
        arch_name = sly.fs.get_file_name(ann_name).split('_')[1] + archive_ext
        arch_path = os.path.join(input_dir, arch_name)
        if not sly.fs.file_exists(arch_path):
            logger.warn('There is no archive {} in the input data, but it must be'.format(arch_name))
            continue

        if zipfile.is_zipfile(arch_path):
            with zipfile.ZipFile(arch_path, 'r') as archive:
                archive.extractall(input_dir)
        else:
            raise Exception("No such file".format(archive_path))

        imgs_dir_path = os.path.join(input_dir, sly.fs.get_file_name(arch_name))
        ann_json = sly.json.load_json_file(ann_path)

        videos = ann_json['videos']
        ovis_anns = ann_json['annotations']
        if not ovis_anns:
            logger.warn('There is no annotations data in {}'.format(ann_name))
            continue

        for category in ann_json['categories']:
            if category['id'] not in ovis_classes.keys():
                ovis_classes[category['id']] = category['name']
                id_to_obj_classes[category['id']] = sly.ObjClass(category['name'], sly.Bitmap)
            else:
                if ovis_classes[category['id']] != category['name']:
                    logger.warn(
                        'Category with id {} corresponds to the value {}, not {}. Check your input annotations'.format(
                            category['id'], ovis_classes[category['id']], category['name']))

        new_dataset = api.dataset.create(new_project.id, sly.fs.get_file_name(ann_name), change_name_if_conflict=True)
        new_meta = sly.ProjectMeta(sly.ObjClassCollection(list(id_to_obj_classes.values())))
        meta = meta.merge(new_meta)
        api.project.update_meta(new_project.id, meta.to_json())

        anns = defaultdict(list)
        for ovis_ann in ovis_anns:
            anns[ovis_ann['video_id']].append([ovis_ann['category_id'], ovis_ann['id'], ovis_ann['segmentations']])

        for video_data in videos:
            no_image = False
            curr_anns = anns[video_data['id']]
            video_objects = {}
            for curr_ann in curr_anns:
                video_objects[curr_ann[1]] = sly.VideoObject(id_to_obj_classes[curr_ann[0]])

            #=============================create video===============================================================
            img_size = (video_data['width'], video_data['height'])
            video_folder = video_data['file_names'][0].split('/')[0]
            video_name = video_folder + video_ext
            images_path = os.path.join(imgs_dir_path, video_folder)
            if not sly.fs.dir_exists(images_path):
                logger.warn('There is no folder {} in the input data, but it is in annotation'.format(images_path))
                continue
            images = os.listdir(images_path)
            progress = sly.Progress('Create video', len(videos), app_logger)
            video_path = os.path.join(extract_dir, video_name)
            video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MP4V'), frame_rate, img_size)
            for curr_ovis_im_path in video_data['file_names']:
                curr_im_path = os.path.join(imgs_dir_path, curr_ovis_im_path)
                if not sly.fs.file_exists(curr_im_path):
                    logger.warn('There is no image {} in {} folder, but it must be. Video will be skipped.'.format(
                        curr_ovis_im_path.split('/')[1], video_name))
                    no_image = True
                    break
                video.write(cv2.imread(curr_im_path))
            if no_image:
                continue
            progress.iter_done_report()
            video.release()
            # =======================================================================================================
            frames = []
            for idx in range(len(images)):
                figures = []
                for fig_id, curr_ann in enumerate(curr_anns):
                    ovis_geom = curr_ann[2][idx]
                    if ovis_geom:
                        mask = decode(ovis_geom).astype(bool)
                        if img_size[1] % 2 == 1:
                            mask[mask.shape[0] - 1, :] = False
                        if img_size[0] % 2 == 1:
                            mask[:, mask.shape[1] - 1] = False
                        geom = sly.Bitmap(mask)
                        figure = sly.VideoFigure(video_objects[curr_ann[1]], geom, idx)
                        figures.append(figure)
                new_frame = sly.Frame(idx, figures)
                frames.append(new_frame)

            file_info = api.video.upload_paths(new_dataset.id, [video_name], [video_path])
            new_frames_collection = sly.FrameCollection(frames)
            new_objects = sly.VideoObjectCollection(list(video_objects.values()))
            if random.random() < samplePercent:
                tag = VideoTag(tag_meta_val)
            else:
                tag = VideoTag(tag_meta_train)
            tag_collection = VideoTagCollection([tag])
            ann = sly.VideoAnnotation((img_size[1], img_size[0]), len(frames), objects=new_objects,
                                      frames=new_frames_collection, tags=tag_collection)
            logger.info('Create annotation for video {}'.format(video_name))
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

