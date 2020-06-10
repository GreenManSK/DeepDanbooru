import os
import sys
from typing import Any, Iterable, List, Tuple, Union

import six
import tensorflow as tf

import deepdanbooru as dd


def evaluate_image(
    image_input: Union[str, six.BytesIO], model: Any, tags: List[str], threshold: float
) -> Iterable[Tuple[str, float]]:
    width = model.input_shape[2]
    height = model.input_shape[1]

    image = dd.data.load_image_for_evaluate(
        image_input, width=width, height=height)

    image_shape = image.shape
    image = image.reshape(
        (1, image_shape[0], image_shape[1], image_shape[2]))
    y = model.predict(image)[0]

    result_dict = {}

    for i, tag in enumerate(tags):
        result_dict[tag] = y[i]

    for tag in tags:
        if result_dict[tag] >= threshold:
            yield tag, result_dict[tag]


def evaluate(target_paths, project_path, model_path, tags_path, threshold, allow_gpu, compile_model, allow_folder, folder_filters, verbose, non_recursive_folders):
    if not allow_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if not model_path and not project_path:
        raise Exception('You must provide project path or model path.')

    if not tags_path and not project_path:
        raise Exception('You must provide project path or tags path.')

    error_log = open("errors.log", "w")

    target_image_paths = []

    for target_path in target_paths:
        if allow_folder and not os.path.isfile(target_path):
            target_image_paths.extend(dd.io.get_image_file_paths_recursive(target_path, folder_filters) if not non_recursive_folders else dd.io.get_image_file_paths(target_path, folder_filters))
        else:
            target_image_paths.append(target_path)

    target_image_paths = dd.extra.natural_sorted(target_image_paths)

    if model_path:
        if verbose:
            print(f'Loading model from {model_path} ...')
        model = tf.keras.models.load_model(model_path, compile=compile_model)
    else:
        if verbose:
            print(f'Loading model from project {project_path} ...')
        model = dd.project.load_model_from_project(project_path, compile_model=compile_model)

    if tags_path:
        if verbose:
            print(f'Loading tags from {tags_path} ...')
        tags = dd.data.load_tags(tags_path)
    else:
        if verbose:
            print(f'Loading tags from project {project_path} ...')
        tags = dd.project.load_tags_from_project(project_path)

    for image_path in target_image_paths:
        print(f'Tags of {image_path}:')
        try:
            for tag, score in evaluate_image(image_path, model, tags, threshold):
                print(f'({score:05.3f}) {tag}')
        except:
            e = sys.exc_info()[0]
            error_log.write("Error with image {0}: {1}\n".format(image_path, str(e)))
            print("ERROR")

        print()
