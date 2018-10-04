This repository contains Adversarial Attacks on CIFAR-10 dataset implemented in Pytorch:
  1) Fast Gradient Sign Method (Untargeted)
  2) Iterative Fast Gradient Sign Method (Untargeted)
  
It will include more Adversarial Attacks and Defenses Technique in future as well

*) The CIFAR-10 Network is trained on VGG-16 architecture based on [1] . It reaches a Test accuracy of 85%

Results :  ( Test Accuracy is evaluated after adding adverserial noise to the test data)

        1) FGSM method:
              a) Epsilon = 0.1  Test Accuracy = 48.26 %
              b) Epsilon = 0.15 Test Accuracy = 43.26 %
              c) Epsilon = 0.2  Test Accuracy = 40.05 %

        2) Iterative FGSM Method:
              a) Epsilon = 0.075
                    i) Iterations = 4   Test Accuracy = 33.06 %
                    ii) Iterations = 10  Test Accuracy = 28.27 %
              b) Epsilon = 0.1
                    i) Iterations = 4    Test Accuracy = 23.52 %
                    ii) Iterations = 10   Test Accuracy = 17.98 %
              c) Epsilon = 0.15
                    i) Iterations = 4     Test Accuracy = 12.7%
                    ii) Iterations = 10   Test Accuract = 6.77 %
                    iii) Iterations = 15  Test Accuracy = 5.93 %
  






References:

[1] Shuying Liu and Weihong Deng. Very deep convolutional neural network based image classifi- cation using small training 
    sample size. In Pattern Recognition (ACPR), 2015 3rd IAPR Asian Conference on, pages 730–734. IEEE, 2015.
 
[2]  Goodfellow, Ian J, Shlens, Jonathon and Szegedy, Christian. "Explaining and harnessing adversarial examples." 
     arXiv preprint arXiv:1412.6572 (2014): . 

