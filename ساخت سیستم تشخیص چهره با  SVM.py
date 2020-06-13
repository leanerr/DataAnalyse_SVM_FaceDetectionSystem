#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)


# با استفاده از یک دیتاست معروف و آموزشی که حاوی چهره های افراد مشهور دنیای سیاست میباشد این سیستم  طراحی شده است

# In[4]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
            xlabel=faces.target_names[faces.target[i]])


# تصاویری و نام هایی که مشاهده میکنید ،از دیتاست بیرون کشیده شده اند و مدل ما سعی خواهد کرد که فرآیند آموزش خود را به کمک این نام ها و تصاویر انجام دهد

# In[9]:


from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import make_pipeline

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)


# In[10]:


from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target,
                                                random_state=42)


# In[11]:


from sklearn.grid_search import GridSearchCV
param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)

get_ipython().run_line_magic('time', 'grid.fit(Xtrain, ytrain)')
print(grid.best_params_)


# In[12]:


model = grid.best_estimator_
yfit = model.predict(Xtest)


# In[13]:


fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);


# سیستم با طی کردن فرآِیند لرنینگ ، نام افراد را با توجه به عکس پیشنهادی سیستم تشخیص میدهد .
# تشخیص اشتباه به رنگ قرمز در خواهد آمد ، همانطور که مشهود است این بار ،تمامی تشخیص ها صحیح بوده اند

# In[14]:


from sklearn.metrics import classification_report
print(classification_report(ytest, yfit,
                            target_names=faces.target_names))


# گزارشی که مشاهده میکنید ، اطلاعات و گزارش مربوط به پیشبینی تشخیص برای نام و تصویر 
# هر یک از افراد است

# # confusion_matrix and seaborn

# In[16]:


import seaborn as sns
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');


# در این ماتریس میتوانید جزییات تشخیص و پیشبینی سیستم طراحی شده و تست شده بر روی داده ها ی تست را مشاهده کنید ، برای مثال : جورج بوش 103 بار به درستی تشخیص داده شده است ، و مثلا 10 بار با آقای کالین پاول اشتباه گرفته شده است.
# 
# آقای شارون 13 بار به درستی  و تنها دو بار با آقای رامسفلد اشتباه گرفته شده است

# گفتنیست که این سیستم جنبه آموزشی و تست داشته است

# علی عسگری
