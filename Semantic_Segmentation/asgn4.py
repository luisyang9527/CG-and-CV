from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from locale import normalize
from operator import concat

import os
import numpy as np
from numpy.core.numeric import extend_all
from numpy.testing._private.utils import requires_memory
import torch
import itertools

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as tf

from matplotlib import pyplot as plt, transforms
from skimage import color
from scipy.sparse import csr_matrix

# from utils import VOC_LABEL2COLOR
# from utils import VOC_STATISTICS
# from utils import numpy2torch
from utils import*
import fnmatch

from torchvision import models

def get_filenames(directory, match='*.*', not_match=()):
    if match is not None:
        if isinstance(match, str):
            match = [match]
    if not_match is not None:
        if isinstance(not_match, str):
            not_match = [not_match]

    result = []
    for dirpath, _, filenames in os.walk(directory):
        filtered_matches = list(itertools.chain.from_iterable([fnmatch.filter(filenames, x) for x in match]))
        filtered_nomatch = list(itertools.chain.from_iterable([fnmatch.filter(filenames, x) for x in not_match]))
        matched = list(set(filtered_matches) - set(filtered_nomatch))
        result += [os.path.join(dirpath, x) for x in matched]
    return result 

def fileparts(directory):
    if os.path.isdir(directory):
        slash_pos = directory.rfind('\\') + 1
        path = directory[:slash_pos]
        subpath = directory[slash_pos:]
        return path, subpath, None
    else:
        slash_pos = directory.rfind('\\') + 1
        path = directory[:slash_pos]
        subpath = directory[slash_pos:]
        basename, ext = os.path.splitext(subpath)
        return path, basename, ext

class VOC2007Dataset(Dataset):
    """
    Class to create a dataset for VOC 2007
    Refer to https://pytorch.org/tutorials/beginner/basics/data_tutorial.html for an instruction on PyTorch datasets.
    """

    def __init__(self, root, train, num_examples):
        super().__init__()
        """
        Initialize the dataset by setting the required attributes.

        Args:
            root: root folder of the dataset (string)
            train: indicates if we want to load training (train=True) or validation data (train=False)
            num_examples: size of the dataset (int)

        Returns (just set the required attributes):
            input_filenames: list of paths to individual images
            target_filenames: list of paths to individual segmentations (corresponding to input_filenames)
            rgb2label: lookup table that maps RGB values to class labels using the constants in VOC_LABEL2COLOR.
        """
        # read basenames to keep
        split = 'train.txt' if train else 'val.txt'
        idx_filname = os.path.join(root, 'ImageSets', 'Segmentation', split)
        basenames = np.loadtxt(idx_filname,dtype=str)[0:num_examples]
        
        # read all filenames
        all_input_filenames = get_filenames(os.path.join(root, 'JPEGImages'), match = '*.jpg')
        all_target_filenames = get_filenames(os.path.join(root, 'SegmentationClass'), match = '*.png')
        
        # filter indices
        input_filenames = sorted(list(filter((lambda fn: fileparts(fn)[1] in basenames), all_input_filenames)))
        target_filenames = sorted(list(filter((lambda fn: fileparts(fn)[1] in basenames), all_target_filenames)))
        print(len(target_filenames))
        # make lookup table
        row_ind = []
        col_ind = []
        data = []
        for i, col in enumerate(VOC_LABEL2COLOR):
            row_ind.append(256*256*col[0] + 256*col[1] + 256*col[2])
            col_ind.append(0)
            data.append(i)
        rgb2label = csr_matrix((data, (row_ind, col_ind)), shape=(256**3, 1), dtype=np.uint8)
        
        self.input_filenames = input_filenames
        self.target_filenames = target_filenames
        self.rgb2label = rgb2label

    def __getitem__(self, index):
        """
        Return an item from the datset.

        Args:
            index: index of the item (Int)

        Returns:
            item: dictionary of the form {'im': the_image, 'gt': the_label}
            with the_image being a torch tensor (3, H, W) (float) and 
            the_label being a torch tensor (1, H, W) (long) and 
        """
         # load image
        im = plt.imread(self.input_filenames[index]).astype(np.float32) / 255.0
        
        # load labels
        mask = (plt.imread(self.target_filenames[index])*255).astype(np.uint8)
        lookup = (256*256)*mask[:,:,0] + 256*mask[:,:,1] + mask[:,:,2]
        label = np.expand_dims(self.rgb2label[lookup.flatten(), :].toarray().reshape(im.shape[0:2]), axis=2)
        print("label={}".format(label.shape))
        
        im_tensor = numpy2torch(im).float()
        label_tensor = numpy2torch(label).long()
        item = {'im': im_tensor, 'gt': label_tensor}
        print("item={}".format(item))

        assert (isinstance(item, dict))
        assert ('im' in item.keys())
        assert ('gt' in item.keys())

        return item

    def __len__(self):
        """
        Return the length of the datset.

        Args:

        Returns:
            length: length of the dataset (int)
        """
        length = len(self.input_filenames)
        return length


def create_loader(dataset, batch_size, shuffle, num_workers=1):
    """
    Return loader object.

    Args:
        dataset: PyTorch Dataset
        batch_size: int
        shuffle: bool
        num_workers: int

    Returns:
        loader: PyTorch DataLoader
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    assert (isinstance(loader, DataLoader))
    return loader


def voc_label2color(np_image, np_label):
    """
    Super-impose labels on a given image using the colors defined in VOC_LABEL2COLOR.

    Args:
        np_image: numpy array (H,W,3) (float)
        np_label: numpy array  (H,W) (int)

    Returns:
        colored: numpy array (H,W,3) (float)
    """
    assert (isinstance(np_image, np.ndarray))
    assert (isinstance(np_label, np.ndarray))

    h, w = np_label.shape
    y = color.rgb2ycbcr(np_image)[:,:,0:1]
    lookup = np.zeros((len(VOC_LABEL2COLOR), 3), dtype = np.uint8)
    
    for i in range(len(VOC_LABEL2COLOR)):
        lookup[i, :] = VOC_LABEL2COLOR[i][:]
    print("lookup.shape={}".format(lookup.shape))
    print("np_label.shape={}".format(np_label.shape))
    print("np_image.shape={}".format(np_image.shape))
    z = lookup[np_label[:], :].reshape(h, w, 3).astype(np.float32) / 255.0
    cbcr = color.rgb2ycbcr(z)[:,:,1:3]
    ycbcr = np.concatenate((y, cbcr), axis = 2)
    colored = np.clip(color.ycbcr2rgb(ycbcr), 0.0, 1.0).astype(np_image.dtype) 

    assert (np.equal(colored.shape, np_image.shape).all())
    assert (np_image.dtype == colored.dtype)
    return colored


def show_dataset_examples(loader, grid_height, grid_width, title):
    """
    Visualize samples from the dataset.

    Args:
        loader: PyTorch DataLoader
        grid_height: int
        grid_width: int
        title: string
    """
    num_images = grid_height * grid_width
    it = iter(loader)
    fig = plt.figure(figsize = (12, 12))
    for i in range(num_images):
        example_dict = next(it)
        im = example_dict['im']
        gt = example_dict['gt']
        im = torch2numpy(im.squeeze(dim = 0))
        gt = torch2numpy(gt.squeeze(dim = 0)).squeeze()
        colorcoding = voc_label2color(im, gt)

        plt.subplot(grid_height, grid_width, i+1)
        plt.imshow(colorcoding)
        plt.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig("%s.pdf" % title, dpi = 200)
    plt.show()

    pass


def normalize_input(input_tensor):
    """
    Normalize a tensor using statistics in VOC_STATISTICS.

    Args:
        input_tensor: torch tensor (B,3,H,W) (float32)
        
    Returns:
        normalized: torch tensor (B,3,H,W) (float32)
    """
    mu = torch.from_numpy(np.array(VOC_STATISTICS['mean'])).reshape(1, 3, 1, 1).float()
    std = torch.from_numpy(np.array(VOC_STATISTICS['std'])).reshape(1, 3, 1, 1).float()
    normalized = (input_tensor - mu) / std

    assert (type(input_tensor) == type(normalized))
    assert (input_tensor.size() == normalized.size())
    return normalized

def run_forward_pass(normalized, model):
    """
    Run forward pass.

    Args:
        normalized: torch tensor (B,3,H,W) (float32)
        model: PyTorch model
        
    Returns:
        prediction: class prediction of the model (B,1,H,W) (int64)
        acts: activations of the model (B,21,H,W) (float 32)
    """
    model.eval()
    acts = model(normalized)['out']
    prediction = torch.argmax(acts, dim = 1, keepdim=True)
    
    assert (isinstance(prediction, torch.Tensor))
    assert (isinstance(acts, torch.Tensor))
    return prediction, acts

def show_inference_examples(loader, model, grid_height, grid_width, title):
    """
    Perform inference and visualize results.

    Args:
        loader: PyTorch DataLoader
        model: PyTorch model
        grid_height: int
        grid_width: int
        title: string
    """
    num_images = grid_height * grid_width
    it = iter(loader)
    fig = plt.figure(figsize=(12, 12))
    for i in range(num_images):
        example_dict = next(it)
        im = example_dict['im']
        gt = example_dict['gt']
        normalized = normalize_input(im)

        with torch.no_grad():
            pred, acts = run_forward_pass(normalized, model)
        avg_prec = average_precision(pred, gt)

        im = torch2numpy(im.squeeze(dim = 0))
        gt = torch2numpy(gt.squeeze(dim = 0)).squeeze()
        res = torch2numpy(pred.squeeze(dim = 0)).squeeze()

        gt_colorcoding = voc_label2color(im, gt)
        res_colorcoding = voc_label2color(im, res)
        concat = np.concatenate((gt_colorcoding, res_colorcoding), axis = 1)

        plt.subplot(grid_height, grid_width, i+1)
        plt.title('avg_prec = %1.2f' % avg_prec)
        plt.imshow(concat)
        plt.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig("inference.pdf", dpi = 200)
    plt.show()

    pass

def average_precision(prediction, gt):
    """
    Compute percentage of correctly labeled pixels.

    Args:
        prediction: torch tensor (B,1,H,W) (int)
        gt: torch tensor (B,1,H,W) (int)
       
    Returns:
        avg_prec: torch scalar (float32)
    """
    nums_correct = (prediction == gt).float().sum()
    num_all = torch.ones_like(prediction).sum()
    avg_prec = nums_correct / num_all

    return avg_prec

### FUNCTIONS FOR PROBLEM 2 ###

def find_unique_example(loader, unique_foreground_label):
    """Returns the first sample containing (only) the given label

    Args:
        loader: dataloader (iterable)
        unique_foreground_label: the label to search

    Returns:
        sample: a dictionary with keys 'im' and 'gt' specifying
                the image sample 
    """
    example = []
    it = iter(loader)
    while True:
        example_dict = next(it)
        gt = example_dict['gt']
        unique_lables = torch.unique(gt)
        if unique_lables.numel() == 2:
            if unique_lables[1] == unique_foreground_label:
                example = example_dict
                break

    assert (isinstance(example, dict))
    return example


def show_unique_example(example_dict, model):
    """Visualise the results produced for a given sample (see Fig. 3).

    Args:
        example_dict: a dict with keys 'gt' and 'im' returned by an instance of VOC2007Dataset
        model: network (nn.Module)
    """
    im = example_dict['im']
    gt = example_dict['gt']

    normalized = normalize_input(im)
    pred, acts = run_forward_pass(normalized, model)
    avg_prec = average_precision(pred, gt)

    im = torch2numpy(im.squeeze(dim = 0))
    gt = torch2numpy(gt.squeeze(dim = 0)).squeeze()
    res = torch2numpy(pred.squeeze(dim = 0)).squeeze()
    
    gt_colorcoding = voc_label2color(im, gt)
    res_colorcoding = voc_label2color(im, res)
    concat = np.concatenate((gt_colorcoding, res_colorcoding), axis = 1)

    plt.figure(figsize=(12, 12))
    plt.imshow(concat)
    plt.axis("off")
    plt.title("avg_prec = %1.2f" % avg_prec)

    plt.tight_layout()
    plt.savefig("cat_before.pdf")
    plt.show()

    pass


def show_attack(example_dict, model, src_label, target_label, learning_rate, iterations):
    """Modify the input image such that the model prediction for label src_label
    changes to target_label.

    Args:
        example_dict: a dict with keys 'gt' and 'im' returned by an instance of VOC2007Dataset
        model: network (nn.Module)
        src_label: the label to change
        target_label: the label to change to
        learning_rate: the learning rate of optimisation steps
        iterations: number of optimisation steps

    This function does not return anything, but instead visualises the results (see Fig. 4).
    """
    model.eval()
    im = example_dict['im']
    gt = example_dict['gt']
    transformed = im.clone()

    # create fake gt label
    fake_gt = gt.clone()
    fake_gt[fake_gt == src_label] = target_label
    fake_gt = fake_gt.squeeze(dim = 1)
    transformed.requires_grad = True
    optimizer = optim.LBFGS([transformed], lr = learning_rate, max_iter=iterations)

    def closure():
        optimizer.zero_grad()
        normalized = normalize_input(transformed)
        pred, acts = run_forward_pass(normalized, model)
        loss = tf.cross_entropy(acts, fake_gt)
        loss.backward()
        transformed.grad = transformed.grad * (fake_gt>0).float()

        print('loss = %4.4f' % loss.item())
        return loss
    
    optimizer.step(closure)
    normalized = normalize_input(transformed.clamp(0, 1))
    transformed_pred, acts = run_forward_pass(normalized, model)
    avg_prec = average_precision(transformed_pred, gt=gt)
    
    diff = torch.norm(im - transformed, p=2, dim=1, keepdim=True)
    diff /= diff.max()
    diff = torch.cat((diff, diff, diff), dim=1)

    im = torch2numpy(im.squeeze(dim = 0))
    transformed = torch2numpy(transformed.detach().squeeze(dim = 0))
    diff = torch2numpy(diff.detach().squeeze(dim = 0))
    transformed = np.clip(transformed, 0.0, 1.0)
    transformed_pred = torch2numpy(transformed_pred.squeeze(dim = 0)).squeeze()
    pred_colorcoding = voc_label2color(transformed, transformed_pred)

    concat1 = np.concatenate((im, transformed), axis=1)
    concat2 = np.concatenate((diff, pred_colorcoding), axis=1)
    concat = np.concatenate((concat1, concat2), axis=0)

    plt.figure(figsize=(12, 12))
    plt.imshow(concat)
    plt.axis("off")
    plt.title('avg_prec = %1.2f' % avg_prec)
    plt.tight_layout()
    plt.savefig("cat_after.pdf")
    plt.show()

    pass


# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
def main():
    # Please set an environment variables 'VOC2007_HOME' pointing to your '../VOCdevkit/VOC2007' folder
    root = os.environ["VOC2007_HOME"]

    # create datasets for training and validation
    train_dataset = VOC2007Dataset(root, train=True, num_examples=128)
    valid_dataset = VOC2007Dataset(root, train=False, num_examples=128)

    # create data loaders for training and validation
    train_loader = create_loader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    valid_loader = create_loader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    # show some images for the training and validation set
    show_dataset_examples(train_loader, grid_height=2, grid_width=3, title='training examples')
    show_dataset_examples(valid_loader, grid_height=2, grid_width=3, title='validation examples')

    # Load FCN network
    model = models.segmentation.fcn_resnet101(pretrained=True, num_classes=21)

    # Apply fcn. Switch to training loader if you want more variety.
    show_inference_examples(valid_loader, model, grid_height=2, grid_width=3, title='inference examples')

    # attack1: convert cat to dog
    cat_example = find_unique_example(valid_loader, unique_foreground_label=23)
    show_unique_example(cat_example, model=model)
    show_attack(cat_example, model, src_label=8, target_label=12, learning_rate=1.0, iterations=10)

    # feel free to try other examples..

if __name__ == '__main__':
    main()
