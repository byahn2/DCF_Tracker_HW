# DCF_Tracker_HW
Given a target, the goal is to estimate the target state over time.  The target is the location of a patch in the first frame of the video, and the output is the position of the target in the following frames.

To do this, we use a Discriminant Correlation Filter (DCF).  A deep neural network extracts the features of the image. In the current frame, there are the features of the target patch and the ideal response which is a gaussian function peaked at the center.  We want to learn the optimal filter that can output the ideal response given the target patch.  We then crop a search patch in the new frame which is an enlarged patch centered at the same position as in the previous frame.  We apply the optimal filter to do cross correlation ont he search patch to generate the correlation response map.  Finally, the target translation can be estimated by searching the maximum value of the correlation response map.  The filter w can be obtained by minimizing the output ridge loss using gradient descent.

## Prepare Dataset

We will use OTB2013 dataset. To get the dataset,

```bash
cd csc249tracking/dataset
python gen_otb2013.py
cd OTB2015
./download.sh 
python unzip.py
cd ..
ln -s $ABSOLUTE_PATH_to_OTB2015 OTB2013
```



To Test, run
```bash
python DCFtracker.py --model classifier_param.pth
```

You can visualize the tracking result by running 

```bash
python DCFtracker.py --model classifier_param.pth --visualization
```

### Compare with standard model

You can check the correctness of `network.py` by running

```shell
python DCFtracker.py
```

it will use the standard tracking model stored in `param.pth`, and the test result (AUC) should be greater than 0.6

