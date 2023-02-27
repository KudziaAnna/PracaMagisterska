# FeTA2022
Fetal Tissue Annotation and Segmentation Challenge (FeTA), MICCAI 2022

****

### Docker manual
**Building docker container**

```docker build -t feta_challenge/sano .```

**Running container**

```docker run -dit -v [TEST-INPUT-IMG]:/input_img/:ro -v [TEST-INPUT-META]:/input_meta/:ro -v /output feta_challenge/sano```

For example:

```docker run -dit --name fetasano -v C:\projects\FeTA\example\example_img\:/input_img/:ro -v C:\projects\FeTA\example\example_img\:/input_meta/:ro -v /output feta_challenge/Sano```

**Inferencing**

```docker exec fetasano python inference.py```

**Copy the output**

```docker cp fetasano:/output [RESULT-TEAM]```

For example:
```docker cp fetasano:/output C:\projects\FeTA\example\output\```

**Stop and remove docker container**

```docker stop fetasano```
```docker rm -v fetasano```