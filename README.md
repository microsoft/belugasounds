![header image](https://github.com/microsoft/belugasounds/blob/master/belugawhale.JPG)


# Introduction

In the U.S., there are five populations of beluga whales, all in Alaska. Of those five, the Cook Inlet population is the smallest and has declined by about seventy-five percent since 1979. This population was listed as an endangered species in 2008, with hopes that the population would begin to recover in the near future, but more than a decade later they continue to decline, with a current population estimate of 328 whales. Read more detailed stories about Cook Inlet beluga whales here: https://www.fisheries.noaa.gov/video/species-spotlight-cook-inlet-beluga-whale

This project is collaborative work among AI for Earth team at Microsoft, AI for Good Research Lab at Microsoft, and NOAA (National Oceanographic and Atmospheric Administration) Alaska Fisheries Science Center. The goal is to build a machine learning model that automatically detects Cook Inlet beluga whale acoustic signals. 


# Data

The original raw data was collected with hydrophones (i.e., underwater microphones) placed in permanent moorings within the Cook Inlet beluga whale critical habitat. Recording datasets included in this study correspond to 5 to 7 month mooring deployments for the ice-free water season (May to September) or winter season (October to April) in 2017-2018 in seven locations, which account for more than 13,000 hours of audio recordings. The NOAA team ran all of these audio recordings through the basic detector that they have used throughout this project, and the results were manually validated. Every detection was labeled as either a true detection (i.e., with beluga whale calls) or a false detection (i.e., without beluga whale calls). This labeled dataset served as training and test data for our machine learning work.


# Methodology

We extracted spectrograms for each corresponding detection, and use them along with the associated labels as input of the binary classification model. Four individual deep learning CNN models were trained (see below), as well as their ensemble model.

•	Model 1: Built a CNN from scratch using AlexNet architecture.

•	Model 2: Transfer learning with fine-tuning from a pre-trained VGG16 model.

•	Model 3: Transfer learning with fine-tuning from a pre-trained ResNet50 model.

•	Model 4: Transfer learning with fine-tuning from a pre-trained DenseNet model.


# Run Scripts

There are 

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
