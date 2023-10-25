![HEART](https://miro.medium.com/v2/resize:fit:720/format:webp/0*g2HrJg3uosek8zF-.jpg)

# _**Heart Disease Prediction Using Machine Learning**_
The main purpose of designing this system is to predict the risk of heart disease in a patient and trying to. we have used Various Machine Learning algorithms to evaluate and train the model, such as Logistic Regression, Decision Tree Classifier, Random Forest Classifier and Support Vector Machine (SVM). The main objective of this project is not just predicting the risk of a heart attack but also analysing and putting some results which might help us to determine the risk factors based on the patient’s vital signs and to extract some insights which helps us to understand their working principle.

# _**Base Paper**_
+ https://www.researchgate.net/publication/351763446_Heart_Disease_Prediction_Using_Machine_Learning
+ https://www.researchgate.net/publication/346432379_Heart_Disease_Detection_Using_Machine_Learning

# _**Algorithm Description**_

**Random Forest Classifier:**
Random Forest Classifier is an ensemble algorithm which works with multiple algorithms parallelly. This is a supervised algorithm and it can be used with both classification and regression problems. The output of the new data is estimated either by using majority voting or average voting technique. Since the algorithm works with bagging technique, multiple decision trees are used to provide the output for the specific input. This is a key difference between decision trees and random forests. While decision trees consider all the possible feature splits, random forests only select a subset of those features. Random forest works best with large datasets and high dimensional.


![RandomForestClassifier](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*hmtbIgxoflflJqMJ_UHwXw.jpeg)


**References**
+ https://www.geeksforgeeks.org/random-forest-regression-in-python/



**Decision Tree Classifier:**
A decision tree is a tool for making decisions and the process for making decisions is in a tree like structure, decision tree is a supervised machine learning algorithm mainly used for predicting the outcome after computing all the attributes.The process flow of Decision tree goes from Root node to leave node i.e., the decision node.


![DecisionTreeClassifier](https://miro.medium.com/v2/resize:fit:720/format:webp/1*xGsYc6aXehD7lyoLEn-mMA.png)


**References**
+ https://www.ibm.com/topics/decision-trees
+ https://www.mastersindatascience.org/learning/introduction-to-machine-learning-algorithms/decision-tree/



**Logistic Regression:**
Logistic Regression is a Supervised algorithm which mostly works in the case of binary classification problems. Logistic regression is a sophisticated algorithm where the data to be trained using this algorithm should be properly presented i.e., Normalized/Scaled, Columns should be Converted to numerical and data should be neat and clean. The output is presented in the form of logit score, where this helps us to predict the likelihood of an event occurring of a given problem. The main reason of getting a S curve in the below chart is that the sigmoid function does the trick of converting the given number in the range between 0 and 1.


![LogisticRegression](https://miro.medium.com/v2/resize:fit:640/format:webp/1*eDeJCcodhj72njIo0x5j0A.jpeg)


**References**
+ https://www.geeksforgeeks.org/understanding-logistic-regression/
+ https://www.ibm.com/topics/logistic-regression



**Support Vector Machine:**
Support vector machines are basically a supervised learning algorithm which classifies the data points by drawing a linear curve and a non-linear curve depending on the data it is dealing with. The boundary that separated the 2 or more classes is called as a hyperplane, though there is a possibility of having some million hyperplanes for our data, but we need to find the hyperplane with maximum margin from all the training points, which makes the algorithm more efficient while predicting on new dataset, it can easily classify on which side the new data belongs to.


![SupportVectorMachine](https://miro.medium.com/v2/resize:fit:640/format:webp/1*-4FOSXGyV6CSUOrOPEFc9g.png)


**References**
+ https://www.geeksforgeeks.org/support-vector-machine-algorithm/

![Happy](https://media0.giphy.com/media/l1J9vjZgVNYsSTTeo/giphy.gif)

# _**How to Execute?**_
So, before execution we have some pre-requisites that we need to download or install i.e., anaconda environment, python and a code editor.
**Anaconda**: Anaconda is like a package of libraries and offers a great deal of information which allows a data engineer to create multiple environments and install required libraries easy and neat.

**Download link:**

![Anaconda](https://1.bp.blogspot.com/-UJ1Ws2zZ9V4/TtMbG2ynJiI/AAAAAAAABbM/m6t2kuEhKdY/s1600/The-biggest-anaconda-snake-3.jpg)

https://www.anaconda.com/

**Python**: Python is a most popular interpreter programming language, which is used in almost every field. Its syntax is very similar to English language and even children and learning it nowadays, due to its readability and easy syntax and large community of users to help you whenever you face any issues.

**Download link:**

![Python](https://i0.wp.com/reptileworldfacts.com/wp-content/uploads/2019/05/male-blonde-super-tiger-reticulated-python.jpg?resize=351%2C351&ssl=1)

https://www.python.org/downloads/

**Code editor**: Code editor is like a notepad for a programming language which allows user to write, run and execute program which we have written. Along with these some code editors also allows us to debug, which usually allows users to execute the code line by line and allows them to see where and how to solve the errors. But I personally feel visual code is very good to work with any programming language and makes a great deal of attachment with user.

**Download links:**

![Vs code](https://schwabencode.com/contents/logos/VS2019-Badge.png) ![Pycharm](https://i0.wp.com/scracked.com/wp-content/uploads/2020/01/PyCharm-2019.3.4-Crack.png?fit=200%2C200&ssl=1)

+ https://code.visualstudio.com/Download, 
+ https://www.jetbrains.com/pycharm/download/#section=windows

# _**How to create a new environment and configure jupyter notebook with it.**_
Let us define an environment and why we need different environments. An environment is a collection of libraries that are required to run our project. When we already have an environment with the necessary libraries, why do we need a new environment?
To avoid version mismatches, we create a new environment for each project. For example, in your previous project, you used "tf env" with tensorflow 2.4 and keras 2.4, but in your current project, you must use tensorflow 2.6 and keras 2.6. If you continue your project in the "tf env" environment, there will be a version mismatch and you will need to update tensorflow and keras, but this will cause problems with the previous project's execution. To avoid this, we create a new environment with tensorflow 2.6 and keras 2.6 and resume our project.

Let us now see how to create an environment in anaconda.
+ Type “conda create –n <<name_of_your_env>>”
example: conda create -n env
+ It will ask to proceed with the environment location, type ‘y’ and press enter.
+ When you press ‘y’, the environment will be created. To activate your environment type conda activate <<your_env_name>> . E.g., conda activate myenv.
+ You can see that the environment got changed after conda activate myenv line. It changed from “base” to “myenv” which means you are now working in “myenv” environment.
+ To install a library in your virtual environment type pip install <library_name>.
e.g., pip install pandas
+ Instead of installing libraries one by one you can even install by bunch, i.e., we have a txt file called requirements.tx which consists of all the libraries required to proceed with the project, so we can use it.
+ so, before installing requirements.txt, make sure you are in the specific path where your requirements.txt is located, basically this file is located in the folder where our executable files are located, so we need to move to that directory by following command.
**cd C:\folder_name**
+ Here A -> drive, folder name -> path where your executable file is saved
+ I go to that file path in anaconda using cd command 
1.	Go to drive where your project file is.
2.	Go to the path of your project using cd <path>
3.	Type pip install –r requirements.txt 
+ And all your required libraries will be downloaded and you can start your project.
+ But if you want to use jupyter notebook on the new environment you have to set it up for the new environment.
+ After you have installed all the libraries and created an environment, you need an editor to run the code, that is starting jupyter notebook, as soon as you enter jupyter notebook in the terminal you will definitely get this error. “Jupiter” is not recognized as an internal or external command.
So, to solve it it we have 2 commands.
1.	conda install –c conda-forge jupyterlab
2.	conda install –c anaconda python
Now you are ready to use jupyter on this environment and start with your project!


![thanks](http://gifimage.net/wp-content/uploads/2017/11/funny-thank-you-gif-12.gif)

# _**Steps to Run the code.**_
**Note:** Make sure you have added path while installing the software’s.

1. Install the prerequisites/software’s required to execute the code.
2. Press windows key and type in anaconda prompt a terminal opens up.
3. Before executing the code, we need to create a specific environment which allows us to install the required libraries necessary for our project.
•	Type conda create -name “env_name”, e.g.: conda create -name project_1
•	Type conda activate “env_name, e.g.: conda activate project_1
4.	Make sure you are in the correct path in your terminal, where you have saved your executable file/folder. E.g.: cd A:\project\AI\Completed\project_name, then press enter.
5. Install necessary libraries from requirements.txt file provided.
6. Run pip install -r requirements.txt or conda install requirements.txt (Requirements.txt is a text file consisting of all the necessary libraries required for executing this python file. If it gives any error while installing libraries, you might need to install them individually.)
7. Run yolo_face.py in your anaconda terminal and make sure to change the path where your executable files are located.
Example: python yoloface.py --image samples/demo.jpg --output-dir outputs/
**Please follow the above links on how to install and set up anaconda environment to execute files.**

# _**Data Description**_
The Datafile which was used in this project was some yolov3 configuration files, A configuration or .cfg file is nothing but a detailed explanation about the number of parameters being used in the model. A typical .cfg files consists of all the parameters required for training and test our model. Some of the parameters such as, Learning rate, convolutional filters, mask, number of classes etc.

# _**Issues Faced.**_
1. We might face an issue while installing specific libraries.
2. Make sure you have the latest version of python, since sometimes it might cause version mismatch.
3. Adding path to environment variables in order to run python files and anaconda environment in code editor, specifically in visual studio code.

**Note: All the required data hasn't been provided over here i.e the model weigths are not provided over here. Please feel free to contact me for any issues.**

### _**Let’s Connect**_
https://www.linkedin.com/in/mudassiruddin21/

![Connect](https://media1.tenor.com/images/888de7ec66dd5053c46d4dba5b415003/tenor.gif?itemid=3455710)

# _**Yes, you now have more knowledge than yesterday, Keep Going.**_
![Happy](https://media1.tenor.com/images/097e8649aeff1e44465d1baa6747cddb/tenor.gif?itemid=5706107)
  
# Heart-Disease-Detection
