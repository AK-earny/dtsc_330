# dtsc_330

HW #3
Using heart rate (hr) and motion sensor data (acc_x, acc_y, acc_z) from the first 3 participants, the classifier achieved an accuracy of 87% on predicting sleep vs. wake (is_sleep).
I chose heart rate and motion features because they directly measure physical activity. Heart rate tends to drop during sleep, and motion sensors capture periods of rest or movement, making these features highly informative for distinguishing sleep from wake states. I had trouble trying to get the files to work the way I needed so I had some assistance from AI, apologies for that. Likely couldve figured it out on my own if
I hadn't waied so long.


HW #4 
XGBoost achieved an accuracy of 0.84, while random forest acheived a 0.78, meaning performance of the model by approximately 6 percentage points.

HW #5 
I would probably ignore names since they are a little unreliable, meaning I would merge phonebook 1 and phonebook 2 on phone number and address. This would require an inner join merge so that you only compare records that match on those keys in both phone books.

HW #9
Using the website, I made an AI that could tell between a cat and a dog. I found dog and cat pictures from the web, 7 and 5 respectively, and created a dog class and a cat class. It was honestly shocking how well it could discern between the two dispite how little data I put in. With my webcam I showed the program my two dogs, which it was 100% certain were dogs, and since I sadly have no cats I showed it pictures of cats which it again was 100% certain were cats. Now when no obvious cats or dogs were on the screen it was about 70% certain I was a dog, likely due to the fact it had more data on dogs than cats. I'm assuming that the NN already had a bunch of images it was trained on, so when I specifically asked for dogs and cats it then tagged all the images that looked like my inputed data as dog/cat? Actually I think it tagged ALL of its images as either dog or cat. I'm not exactly 100% sure of that though.