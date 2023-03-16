This repository is a log for my NTU Final Year Projectw where I studied about Incremental Training for Watermarked Neural Networks.

For a detailed explanation of my study, please refer to the attached report.

## Background
The demand for high-performance neural networks opened the floodgates to a new market of Ma-
chine Learning as a Service (MLaaS) [4] . Companies of this trending market would provide services
to train and tune models according to the clients’ needs at a negligible cost, as compared to the
cost of infrastructure and preparation needed to train their own models. As the availability of data
increases over time, clients have the option to further fine-tune the models for increased accuracy, or
even perform Transfer Learning to solve similar machine learning problems [5]. This makes MLaaS
a very viable purchase for small and medium sized businesses that lack the sophisticated hardware
to train models. The biggest concerns of MLaaS are the legality and security issues behind the
neural network models, specifically pertaining to intellectual property rights (IPR) [6]. Examples
include clients of the services redistributing the models outside of the contractual agreement, or
even selling the models to other customers, hence directly threatening the business. Intuitively,
it is necessary to develop a robust safeguard mechanism to protect the IPR of such businesses.
Extensive research has been done to develop digital watermarking methods in efforts of preserving
IPR. Some notable studies include ROWBACK, a watermarking scheme which robustness leverages
on adversarial samples in the Trigger Set and the uniform distribution of backdoors throughout the
layers in the neural networks

## Motivation
The robustness of watermarking schemes such as ROWBACK [7], Randomized Smoothing [8] and
BlackMarks [9] have been proven against notable watermark removal attacks like Re-markable
[10], but much uncertainty exists on how they are affected by Incremental Training. Incremental
Training refers to enriching models with newly acquired training data in efforts of improving model
performance. The reliability of robust watermarking schemes largely depends on their verification
through the Trigger Set. The event of having watermarks being unintentionally removed during
Incremental Training would greatly undermine the scheme’s ability to protect rightful ownership.
Hence, it is of utmost importance to verify their susceptibility to Incremental Training.

## Objective and scope
This study aims to investigate on various existing watermarking scheme’s ability to maintain veri-
fiable, retaining IPR and robustness against adversaries, even after Incremental Training. We will
also attempt to discover how certain variables such as Trigger Set Type and learning rates within
such watermarking schemes would affect Incremental Training.
In this paper, we utilized the CIFAR-10 Dataset in our experiments with the various watermark-
ing schemes. We will first split the dataset into different portions such as Train Set, Incremental Set
and Test Set. We will then train a ResNet model [11] with Train Set and generate the appropriate
Trigger Set, e.g. randomly mislabelled samples, adversarial samples [12], unrelated samples, etc.
The model will then be embedded with the watermarks and tested for the Trigger Set Accuracy.
It is key that at this point, the Trigger Set Accuracy recorded is of high value before the actual
experimentation. Finally, we will commence Incremental Training and study the results for the
different watermarking schemes.
The findings will assist in developing a framework for a robust watermarking scheme for neural
networks, with ability to support Incremental Training, ultimately preserving IPR.




## Hardware Specifications


• CPU: 12th Gen Intel(R) Core(TM) i7-12700


• RAM: 32GB

• GPU: NVIDIA RTX A4000

## Libraries

• pandas: 1.51

• jupyterlab: 3.5.0

• ipykernel: 6.17.0

• tensorflow: 2.10.0

• torchvision: 0.14.0

• Pillow: 9.3.0

