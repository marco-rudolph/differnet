# DifferNet

This is the official repository to the WACV 2021 paper "[Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows](
https://arxiv.org/abs/2008.12577)" by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.

If the only reason you ended up here is because you made a typo on 'different' - what was our intention - here is a shortened summary: We introduce a method that is able to find anomalies like defects on image data without having some of them in the training set.


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/same-same-but-differnet-semi-supervised/anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad?p=same-same-but-differnet-semi-supervised)

## Getting Started

You will need [Python 3.6](https://www.python.org/downloads) and the packages specified in _requirements.txt_.
We recommend setting up a [virtual environment with pip](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
and installing the packages there.

Install packages with:

```
$ pip install -r requirements.txt
```

Or install with for Windows as per [PyTorch official site](https://pytorch.org/get-started/locally/):

```
$ pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_s
table.html
$ pip install -r requirements.txt
```

## Configure and Run

All configurations concerning data, model, training, visualization etc. can be made in _config.py_. The default configuration will run a training with paper-given parameters on the provided dummy dataset. This dataset contains images of 4 squares as normal examples and 4 circles as anomaly.

If you encounter GPU Out of Memory issue, you can reduce the neuron numbers in _config.py_
```
fc_internal = 1536 # number of neurons in hidden layers of s-t-networks
```

To start the training, just run _main.py_ as follows! If training on the dummy data does not lead to an AUROC of 1.0, something seems to be wrong.
Please report us if you have issues when using the code.

```
$ python main.py
```

## Data
How to use Data extraction tool to extract data from video clips:
 1. Create folder structure like the example shows in the picture below.
 
  ![1](https://github.com/zerobox-ai/differnet/blob/zijian/dataset/data-generation/annotations/structure1.png)
  
 2. Dump the videos and annotations (rename them use 1.xml, 1.avi as one pair annotation and video) into the folders under data-generation folder.
 
  ![2](https://github.com/zerobox-ai/differnet/blob/zijian/dataset/data-generation/annotations/structure2.png)
  
 3. Modify the annotation files: Since the annotation uses label "defect" to indicate the defect area, while, both good and defective bottles are labeled as "bottle" which is confusing. To indicate which "bottle" is defective, we need to find the frames that labeled with defect, and then manully update the group's label from "bottle" to "defective" for the groups that falling in to those frames. 
  ![3](https://github.com/zerobox-ai/differnet/blob/zijian/dataset/data-generation/annotations/structure3.png)
  
 - For example: in the example image above, the frame 15 and 16 are labeled as "defect" which indicates those 2 frames has defect areas on the bottles. So we need to find the group that contains frame 15 and 16, and then manully update the label from "bottle" to "defective". and then delete the whole \<track\> group that labeled as "defect" (since we don't care about the defect area in data extraction).
 
 4. Modify the config.py, fill in appropriate value for num_videos, save_cropped_image_to and save_original_image_to
 
 5. run the data extraction: python data_extraction.py


The given dummy dataset shows how the implementation expects the construction of a dataset. Coincidentally, the [MVTec AD dataset](https://www.mvtec.com/de/unternehmen/forschung/datasets/mvtec-ad/) is constructed in this way.

Set the variables _dataset_path_ and _class_name_ in _config.py_ to run experiments on a dataset of your choice. The expected structure of the data is as follows:

``` 
train data:

        dataset_path/class_name/train/good/any_filename.png
        dataset_path/class_name/train/good/another_filename.tif
        dataset_path/class_name/train/good/xyz.png
        [...]

test data:

    'normal data' = non-anomalies

        dataset_path/class_name/test/good/name_the_file_as_you_like_as_long_as_there_is_an_image_extension.webp
        dataset_path/class_name/test/good/did_you_know_the_image_extension_webp?.png
        dataset_path/class_name/test/good/did_you_know_that_filenames_may_contain_question_marks????.png
        dataset_path/class_name/test/good/dont_know_how_it_is_with_windows.png
        dataset_path/class_name/test/good/just_dont_use_windows_for_this.png
        [...]

    anomalies - assume there are anomaly classes 'crack' and 'curved'

        dataset_path/class_name/test/crack/dat_crack_damn.png
        dataset_path/class_name/test/crack/let_it_crack.png
        dataset_path/class_name/test/crack/writing_docs_is_fun.png
        [...]

        dataset_path/class_name/test/curved/wont_make_a_difference_if_you_put_all_anomalies_in_one_class.png
        dataset_path/class_name/test/curved/but_this_code_is_practicable_for_the_mvtec_dataset.png
        [...]
``` 

## Credits

Some code of the [FrEIA framework](https://github.com/VLL-HD/FrEIA) was used for the implementation of Normalizing Flows. Follow [their tutorial](https://github.com/VLL-HD/FrEIA) if you need more documentation about it.


## Citation
Please cite our paper in your publications if it helps your research. Even if it does not, you are welcome to cite us.

    @inproceedings { RudWan2021,
    author = {Marco Rudolph and Bastian Wandt and Bodo Rosenhahn},
    title = {Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows},
    booktitle = {Winter Conference on Applications of Computer Vision (WACV)},
    year = {2021},
    month = jan
    }
    
Another paper link because you missed the first one:

* [Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows](
https://arxiv.org/abs/2008.12577)

## License

This project is licensed under the MIT License.

 
