# # # # # # # name = 'Alice'
# # # # # # # age = 25
# # # # # # # height = 170
# # # # # # # is_student = True

# # # # # # # print('Hi, my name is ', name, 'Im ',  age , 'years old. ', 'Student: ', is_student )

# # # # # # # foods = ["ad", "ab", "abd" ]
# # # # # # # for food in foods:
# # # # # # #     print("i like", food)
    
# # # # # # # def greet(name):
# # # # # # #     print("hello",  name)
    
# # # # # # # greet("Elle")
# # # # # # # greet("Jae")

# # # # # # # age = 11
# # # # # # # if age>=18:
# # # # # # #     print("youre an adult!")
# # # # # # # else:
# # # # # # #     print("youre a minor")
    
    
    
# # # # # # # while True:
# # # # # # #     score = 0



# # # # # # #     # Check each answer
# # # # # # #     answer1 = input("What is the capital of Korea? ")
# # # # # # #     if answer1.lower() == "seoul":
# # # # # # #         print("Correct!")
# # # # # # #         score += 1
# # # # # # #     else:
# # # # # # #         print("Wrong! The correct answer is Seoul.")

# # # # # # #     answer2 = input("What is the capital of America? ")
# # # # # # #     if answer2.lower() == "washington" or answer2.lower() == "washington dc":
# # # # # # #         print("Genius!")
# # # # # # #         score += 1
# # # # # # #     else:
# # # # # # #         print("Wrong! The correct answer is Washington, D.C.")

# # # # # # #     answer3 = input("What is the capital of England? ")
# # # # # # #     if answer3.lower() == "london":
# # # # # # #         print("Nice!!")
# # # # # # #         score += 1
# # # # # # #     else:
# # # # # # #         print("Wrong! The correct answer is London.")
        
# # # # # # #     answer4 = input("What is the capital of Japan?")
# # # # # # #     if answer4.lower() == "tokyo":
# # # # # # #         print("whatever")
# # # # # # #         score +=1


# # # # # # #     entry = "youre score is,"
# # # # # # #     if(score == 3):
# # # # # # #         print(entry, "Awesome")
# # # # # # #     elif(score==2):
# # # # # # #         print(entry, "Not bad!")
# # # # # # #     else:
# # # # # # #         print(entry, "terrible!! keep practicing!!!!!!")


# # # # # # #     play_again = input("Do you want to play again? (y/n)")

# # # # # # #     if play_again.lower() != "y":
# # # # # # #         print("thanks for playing ill see you next time!")
# # # # # # #         break












# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================


# # # # # # import random
# # # # # # import os

# # # # # # class QuizGame:
# # # # # #     def __init__(self):
# # # # # #         self.questions = [
# # # # # #             {"question": "What is the capital of Korea?", "answer": "seoul"},
# # # # # #             {"question": "What is the capital of USA?", "answer": "washington"},
# # # # # #             {"question": "What is the capital of Estonia?", "answer": "Tallin"},
# # # # # #             {"question": "What is the capital of England?", "answer": "London"},
# # # # # #             {"question": "What is the capital of Japan?", "answer": "tokyo"},
# # # # # #             {"question": "What is the capital of Canada?", "answer": "ottawa"},
# # # # # #             {"question": "What is the capital of Italy?", "answer": "rome"},
# # # # # #             {"question": "What is the capital of France?", "answer": "paris"},
# # # # # #             {"question": "What is the capital of Spain?", "answer": "Madrid"},
# # # # # #             {"question": "What is the capital of Scotland?", "answer": "Edinburgh"},
# # # # # #             {"question": "What is the capital of Malaysia?", "answer": "KL"},
            
# # # # # #         ]

# # # # # #     #checks if file exists. prevents FileNotFoundError if it doesnt
# # # # # #     #open("highscore.txt", "r") opens the file in read mode. the with statement safely opens the file and ensures it closes automatically
# # # # # #     #f.read() reads the entire content of the file as a string. ex) if file contains 5, this will return  the string "5"
# # # # # #     #int(f.read()) converts the string to an integer
# # # # # #     #return 0 -> if the file doesnt exist, it just returns 0 as the default high score
# # # # # #     def load_high_score():
# # # # # #         if os.path.exists("highscore.txt"):
# # # # # #             with open("highscore.txt", "r") as f:
# # # # # #                 return int(f.read())
# # # # # #         return 0

# # # # # #     def save_high_score(score):
# # # # # #         with open("highscore.txt", "w") as f:
# # # # # #             f.write(str(score))

# # # # # #     # print(len(questions))
    
# # # # # #     def start(self):
# # # # # #         while True:
# # # # # #             self.score = 0
# # # # # #             print("\n Welcome to the capital quiz!!")
            
            
# # # # # #             difficulty = input("Select your difficulty! (hard/medium/easy):").strip().lower()
# # # # # #             self.set_difficulty(difficulty)    
            
# # # # # #             test_type = input("Select a test type(writing/multiple):").strip().lower()

# # # # # #             print(f"current high score: {self.high_score}")
# # # # # #             self.ask_questions(test_type)
            
# # # # # #             self.show_results()
            
# # # # # #             if self.score > self.high_score:
# # # # # #                 print("new high score!!!")
# # # # # #                 self.save_high_score()
# # # # # #                 self.high_score = self.score
                
# # # # # #             again = input("Play again??(y/n): ").strip().lower()
# # # # # #             if again != 'y':
# # # # # #                 print("Thank you for playing!!!")
# # # # # #                 break
            
    
# # # # # #     def set_difficulty(self, difficulty):
# # # # # #         random.shuffle(self.questions)
# # # # # #         if difficulty == 'easy':
# # # # # #             self.quiz_questions = self.questions[:4]
# # # # # #         elif difficulty == 'medium':
# # # # # #             self.quiz_questions = self.questions[:7]
# # # # # #         elif difficulty ==  'hard' :
# # # # # #             self.quiz_questions = self.questions[:11]
# # # # # #         else:
# # # # # #             print("invalid difficulty! proceeding with default difficulty")
# # # # # #             self.quiz_questions = self.questions[:5]
        
    
# # # # # #     def ask_questions(self, test_type):
# # # # # #         for q in self.quiz_questions:
# # # # # #             if test_type == 'writing':
# # # # # #                 # 'q' here is a dictionary representing a single quiz question
# # # # # #                 answer = input(q["question"] + " ").strip().lower()
# # # # # #                 if answer == q["answer"].lower():
# # # # # #                     print("Correct!!!")
# # # # # #                     self.score +=1
# # # # # #                 else:
# # # # # #                     print(f"Wrong!! the answer is {q['answer'].capitalize()}")
# # # # # #             else:
# # # # # #                 self.ask_multiple_choice(q)
    
    
# # # # # #     def ask_multiple_choice(self, question):
# # # # # #         correct_answer = question["answer"]
# # # # # #         wrong_answers = [q["answer"] for q in self.questions if q["answer"] != correct_answer]
# # # # # #         #this part generates 4 choices including 1 correct and 3 wrong answers
# # # # # #         options = random.sample(wrong_answers, 3) + correct_answer
# # # # # #         random.shuffle(options)
# # # # # #         option_letters = ['A','B','C','D']
# # # # # #         #this line pairs up the option letters and the options
# # # # # #         #zip takes two lists and pairs up their elements by position
# # # # # #         #dict function turns those pairs into a dictionary
# # # # # #         option_map = dict(zip(option_letters, options))
# # # # # #         #up there, question is a dictionary like question: answer:
# # # # # #         #this line on the bottom retrieves the text from the question . eg)what is the capital of Korea?
# # # # # #         #\n adds a blank line before the question
        
# # # # # #         print("\n" + question["question"])
        
# # # # # #         for letter in option_letters:
# # # # # #             print(f"{letter}. {option_map[letter]}")
            
# # # # # #         while True:
# # # # # #             choice = input("Your answer: (A,B,C,D)").strip().upper()
# # # # # #             if choice  in option_map:
# # # # # #                 break
# # # # # #             print("Invalid option!!!")
            
# # # # # #         if option_map[choice].lower() == correct_answer.lower():
# # # # # #             print("Correct!!")
# # # # # #             self.score +=1
            
# # # # # #         else:
# # # # # #             print(f"Wrong!! the answer was: {correct_answer}")
            
        
        
# # # # # #     def show_results(self):     
# # # # # #         total = len(self.quiz_questions)      
# # # # # #         print(f"\n you got {self.score}/{total} correct.") 
        
# # # # # #         percent = (self.score/total) * 100
# # # # # #         print(f"Your score is: {percent}")
        
# # # # # #         if percent ==100:
# # # # # #             print("Excellent!! amazing job")
# # # # # #         elif percent >= 80:
# # # # # #             print("good job!! keep it up!")
# # # # # #         elif percent >=60:
# # # # # #             print("you SHOULD STRUDY")
# # # # # #         elif percent >= 40:
# # # # # #             print("Are you dumb???")
# # # # # #         elif percent >= 20 :
# # # # # #             print("You're retarded")
# # # # # #         else:
# # # # # #             print("I have nothing to say. hahaha just give up")
            
            
# # # # # # # Run the game
# # # # # # if __name__ == "__main__":
# # # # # #     game = QuizGame()
# # # # # #     game.start()
            
                
                
# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================
# # # # # # #================================================================================================================================================================================


# # # # # # monry = True 
# # # # # # if monry :
# # # # # #     print("ddd")
# # # # # # else:
# # # # # #     print("dfadfa")
    
    
# # # # # # pocket = ['paper', 'mondet', 'cellphone']
# # # # # # if 'monet' in pocket:
# # # # # #     pass
# # # # # # else:
# # # # # #     print("card out")

# # # # # # treeHit = 0

# # # # # # while treeHit <10:
# # # # # #     treeHit = treeHit + 1
# # # # # #     print("hit the tree %d times." %treeHit)
# # # # # #     if treeHit == 10:
# # # # # #         print("wowww")
        
        
# # # # # # coffee = 10
# # # # # # money = 300
# # # # # # while money:
# # # # # #     print("you paid so ill give you coffee")
# # # # # #     coffee = coffee-1
# # # # # #     print("left coffee is %d" %coffee)
# # # # # #     if coffee ==0:
# # # # # #         break
    
    
# # # # # # bool = 'a' in ('a','b','c')
# # # # # # print(bool)

# # # # # # y = 3.42134234
# # # # # # print(f"{y:0.4f}")

# # # # # # #count
# # # # # # a = "hobby"
# # # # # # print(a.count('b'))

# # # # # # b = "python is the best choice"
# # # # # # print(b.find('b'))

# # # # # # c = "life is too short"
# # # # # # print(c.index('t'))

# # # # # # d= "  hello "
# # # # # # print(d.strip())

# # # # # # e = "life is too short"
# # # # # # f = e.replace("life", "your leg")
# # # # # # print(f)


# # # # # import numpy as np
# # # # # import pandas as pd
# # # # # from tensorflow.keras.models import Sequential
# # # # # from tensorflow.keras.layers import Dense

# # # # # from sklearn.model_selection import train_test_split
# # # # # from sklearn.metrics import r2_score, mean_squared_error

# # # # # path = 'Study25/_data/kaggle/bike/'
# # # # # path_save = 'Study25/_data/kaggle/bike/csv_files/'

# # # # # train_csv= pd.read_csv(path + 'train.csv', index_col = 0)
# # # # # test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
# # # # # submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)

# # # # # print(train_csv.columns)
# # # # # print(test_csv.columns)

# # # # # x = train_csv.drop(['casual', 'registered', 'count'], axis =1)
# # # # # y = train_csv['count']

# # # # # print(f"X shape: {x.shape}")
# # # # # print(f"y shape: {y.shape}")

# # # # # x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=333)

# # # # # model = Sequential()
# # # # # model.add(Dense(128, input_dim = 8, activation = 'relu'))
# # # # # model.add(Dense(64, activation = 'relu'))
# # # # # model.add(Dense(64, activation = 'relu'))
# # # # # model.add(Dense(64))
# # # # # model.add(Dense(16))
# # # # # model.add(Dense(1))


# # # # # model.compile(loss = 'mse', optimizer = 'adam')
# # # # # hist = model.fit(x_train, y_train, epochs= 300, batch_size = 32, validation_split= 0.2, verbose= 1)

# # # # # import matplotlib.pyplot as plt

# # # # # plt.figure(figsize=(9,6))
# # # # # plt.plot(hist.history['loss'], c = 'red', label = 'loss')
# # # # # plt.plot(hist.history['val_loss'], c = 'blue', label='val_loss')
# # # # # plt.title("kaggle bike loss")
# # # # # plt.xlabel("epoch")
# # # # # plt.ylabel('loss')
# # # # # plt.legend(loc = 'upper left')
# # # # # plt.grid()

# # # # # plt.show()

# # # # # #evaluate and predict
# # # # # loss = model.evaluate(x_test, y_test)
# # # # # print("loss:", loss)


# # # # # result = model.predict(x_test)
# # # # # r2= r2_score(y_test, result)
# # # # # print('r2 score:', r2)

# # # # # rmse = np.sqrt(mean_squared_error(y_test, result))

# # # # # print('RMSE:', rmse)

# # # # # y_submit = model.predict(test_csv)
# # # # # print(y_submit.shape)

# # # # # submission_csv['count'] = y_submit
# # # # # print(submission_csv.head())

# # # # # submission_csv.to_csv(path_save + 'submission_33333333.csv')

# # # # import numpy as np
# # # # import pandas as pd
# # # # from tensorflow.keras.models import Sequential
# # # # from tensorflow.keras.layers import Dense
# # # # from sklearn.model_selection import train_test_split

# # # # data_url = "http://lib.stat.cmu.edu/datasets/boston"
# # # # raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# # # # data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# # # # target = raw_df.values[1::2, 2]

# # # # # y data = target data

# # # # #1 data
# # # # # dataset = load_boston()
# # # # # # print(dataset)
# # # # # #Describe
# # # # # print(dataset.DESCR) #(506,13)
# # # # # print(dataset.feature_names)
# # # # # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
# # # # #  'B' 'LSTAT']

# # # # x = data
# # # # print(x.shape)
# # # # y = target
# # # # exit()


# # # # x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.8, random_state= 333)


# # # # # #2 model 

# # # # # model = Sequential()
# # # # # model.add(Dense(64, input_dim = ))


# # # import numpy as np 
# # # import pandas as pd

# # # from tensorflow.keras.models import Sequential
# # # from tensorflow.keras.layers import Dense
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.metrics import r2_score, mean_squared_error
# # # from tensorflow.keras.callbacks import EarlyStopping

# # # path = 'Study25/_data/dacon/ddarung/'


# # # train_csv = pd.read_csv(path + 'train.csv', index_col= 0)

# # # # print(train_csv)
# # # print(train_csv.describe())

# # # print(train_csv.shape)
# # # # print(train_csv.head)
# # # # print(train_csv.columns)
# # # # [1459 rows x 11 columns]

# # # test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
# # # # print(train_csv.info)

# # # submission_csv = pd.read_csv(path + 'submission.csv', index_col = 0)
# # # # print(submission_csv.info)

# # # #gettin rid of the missing values
# # # train_csv = train_csv.dropna()

# # # test_csv = test_csv.fillna(train_csv.mean())

# # # #dropping column count
# # # x= train_csv.drop(['count'], axis = 1)
# # # y = train_csv['count']

# # # x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.8, random_state= 333)

# # # model = Sequential()
# # # model.add(Dense(100, input_dim = 9, activation = 'relu'))
# # # model.add(Dense(100, activation = 'relu'))
# # # model.add(Dense(100, activation = 'relu'))
# # # model.add(Dense(100, activation = 'relu'))
# # # model.add(Dense(100, activation = 'relu'))
# # # model.add(Dense(1))


# # # #compile and fit
# # # model.compile(loss = 'mse', optimizer = 'adam')
# # # es = EarlyStopping(
# # #     monitor = 'val_loss',
# # #     mode = 'min',
# # #     patience = 30,
# # #     restore_best_weights = True,
# # # )

# # # hist = model.fit(x_train, y_train, epochs = 300, batch_size = 16, validation_split = 0,2,
# # #                  callbacks = [es])


# # # #evaluate and predict
# # # loss = model.evaluated(x_test, y_test)
# # # print('loss:', loss)

# # # result = model.predict(x_test)

# # # r2 = r2_score(y_test, result)
# # # print('r2 result:', r2_score)



# # # y_submit = model.predict(test_csv)

# # # submission_csv['count'] = y_submit

# # # submission_csv.to_csv(path + '')






# # import numpy as np
# # import pandas as pd
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layes import Dense
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import r2_score, mean_squared_error
# # from tensorflow.keras.callbacks import EarlyStopping

# # path = 'Study25/_data/kaggle/bike/'
# # path_save = 'Study25/_data/kaggle/bike/csv_files/'

# # train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
# # test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
# # submission_csv = pd.read_csv(path+ 'sampleSubmission.csv', index_col=0)

# # x = train_csv.drop(['casual'], ['registered'], ['count'], axis=1)

# # y = train_csv['count']

# # x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.8, random_state=333)


# # model = Sequential()
# # model.add(Dense(100, input_dim = 8, activation = 'relu'))
# # model.add(Dense(100, activation = 'relu'))
# # model.add(Dense(100, activation = 'relu'))
# # model.add(Dense(100, activation = 'relu'))
# # model.add(Dense(100, activation = 'relu'))
# # model.add(Dense(100, activation = 'relu'))
# # model.add(Dense(1))


# # model.compile(loss = 'mse', optimizer = 'adam')
# # from tensorflow.keras.callbacks import EarlyStopping

# # es = EarlyStopping(
# #     monitor = 'val_loss',
# #     mode = 'min',
# #     patience =30,
# #     restore_best_weights = True,
# # )

# # model.fit(x_train, y_train, epochs =100, batch_size = 2, validation_split = 0.2, 
# #           callbacks = [es])


# # loss = model.evaluate(x_test, y_test)
# # print('loss:', loss)

# # result = model.predict(x_test)


# # y_submit = model.predict(test_csv)
# # submission_csv['count'] = y_submit
# # submission_csv.to_csv(path_save + 'submissionlafjlkdjfs.csv')

# import numpy as np
# import pandas as pd

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# from tensorflow.keras.callbacks import EarlyStopping

# path = 'Study25/_data/kaggle/bike/'
# path_save = 'Study25/_data/kaggle/bike/csv_files/'

# train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
# test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
# submission_csv = pd.read_csv(path+ 'sampleSubmission.csv', index_col=0)

# x = train_csv.drop(['casual', 'registered', 'count'])
# y = train_csv['count']


# x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=333)


# model = Sequential()
# model.add(Dense(100,input_dim = 8, activation = 'relu'))
# model.add(Dense(100, activation = 'relu'))
# model.add(Dense(100, activation = 'relu'))
# model.add(Dense(100, activation = 'relu'))
# model.add(Dense(100, activation = 'relu'))
# model.add(Dense(100, activation = 'relu'))
# model.add(Dense(1))


# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor = 'val_lostt',
#                    mode = 'min',
#                    patience = 30,
#                    restore_best_weights = True)

# model.fit(x_train, y_train, epochs = 100, batch_size = 16, validation_split = 0.2, callbacks = [es])



import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

import time
from sklearn.datasets import load_breast_cancer

#1 data
dataset = load_breast_cancer()
print(dataset.DESCR)


x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state= 333)

print(np.unique(y, return_counts=True))
print(pd.value_counts(y))

exit()

#2 model
model = Sequential()
model.add(Dense())



