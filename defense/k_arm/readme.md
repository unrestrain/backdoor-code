code for the method K-Arm Optimization, the code can run.   
I trained a very easy model which identify that the color of a pure image is white or black. The model used is resnet18 and the acc is more than 97%.I added a backdoor triger into the model. The triger is 3*3 white square in the left up corner. and the attack acc is 100%.Then I detect The backdoor model use k arm optimizer, but it returned a backdoor size 0 which shows that the model don't have backdoor. Now I don't know why. I want to try again with trojai dataset, but I haven't found out about the image data in trojai. 

I test the method with models on trojai webset, finding k-arm can differenciate most backdoor and clean model. So, the reason why my own backdoor model can't be detected need further analysis.
