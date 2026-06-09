#!/usr/bin/env python3
"""
Testing script for NEMAR models.

Usage:
    python test.py --dataroot ./datasets/rire --name my_model --model nemar --phase test
"""

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util.util import tensor2im

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed
    opt.display_id = -1  # no visdom display
    opt.dataset_mode = opt.dataset_mode or 'rire_2d'

    # create dataset
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f'Loading dataset: {dataset_size} images')

    # create model
    model = create_model(opt)
    model.setup(opt)

    # create a directory for saving results
    results_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # create image directory
    images_dir = os.path.join(results_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # test with eval mode
    if opt.eval:
        model.eval()

    print(f'Results will be saved to: {results_dir}')

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images
            break

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        if i % 5 == 0:
            print(f'Processing image {i}/{opt.num_test}: {img_path}')

        # Save images
        for label, image in visuals.items():
            image_numpy = tensor2im(image)
            img_basename = os.path.basename(img_path[0])
            save_name = f'{label}_{os.path.splitext(img_basename)[0]}.png'
            save_path = os.path.join(images_dir, save_name)
            from PIL import Image
            Image.fromarray(image_numpy).save(save_path)

    print(f'Testing completed! Results saved to: {results_dir}')