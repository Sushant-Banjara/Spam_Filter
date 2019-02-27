Spam Text Classification
================

Now that I have selected the model, I now write a script that mines the required text from a .txt file for spam classification and classifies the file as spam or non-spam.The email is a spam email, I had personally received which is as follows

``` r
#load the required libraries
library('RColorBrewer')
library('SnowballC')
library('wordcloud')
library('tm')
library('stringr')
library('devtools')
library('dplyr')
library('FNN')
```

``` r
setwd('C:/Users/spb65/Desktop/Applied Machine Learning')

#read the crude text file 
crude = readLines('New text Document.txt')
#read the training data
train = read.csv('spam_train_imputed.csv', header = TRUE)

#add a row in the train data frame
train[nrow(train)+1,] <- 0

#Convert the text file to corpus
docs = Corpus(VectorSource(crude))
docs1 = Corpus(VectorSource(crude))

#Remove special characters
toSpace = content_transformer(function (x , pattern ) gsub(pattern, " ", x))
toDollar = content_transformer(function (x , pattern ) gsub(pattern, "$", x))
docs = tm_map(docs, toSpace, "/")
docs = tm_map(docs, toSpace, "\\.")
docs = tm_map(docs, toSpace, "@")
docs = tm_map(docs, toSpace, "\\|")
docs = tm_map(docs, toDollar, '???')

# Convert the text to lower case
docs = tm_map(docs, content_transformer(tolower))
# Remove numbers
#docs = tm_map(docs, removeNumbers)
# Remove english common stopwords
docs = tm_map(docs, removeWords, stopwords("english"))
# Remove your own stop word
# specify your stopwords as a character vector
docs = tm_map(docs, removeWords, c("blabla1", "blabla2")) 
# Remove punctuations
docs = tm_map(docs, removePunctuation)
# Eliminate extra white spaces
docs = tm_map(docs, stripWhitespace)


#Convert to data frame 
dtm = TermDocumentMatrix(docs)
mat = as.matrix (dtm)
count = sort (rowSums(mat), decreasing =TRUE)
df = data.frame(count)

#count the total number of words in the email
wordcount = sum(df$count)
```

``` r
#Add the percentage wordcounts in the final row of the data frame
train[nrow(train),1] = (sum(str_count(docs, ' make '))/wordcount)*100
train[nrow(train),2] = (sum(str_count(docs, ' address '))/wordcount)*100
train[nrow(train),3] = (sum(str_count(docs, ' all '))/wordcount)*100
train[nrow(train),4] = (sum(str_count(docs, ' 3d '))/wordcount)*100
train[nrow(train),5] = (sum(str_count(docs, ' our '))/wordcount)*100
train[nrow(train),6] = (sum(str_count(docs, ' over '))/wordcount)*100
train[nrow(train),7] = (sum(str_count(docs, ' remove '))/wordcount)*100
train[nrow(train),8] = (sum(str_count(docs, ' internet '))/wordcount)*100
train[nrow(train),9] = (sum(str_count(docs, ' order '))/wordcount)*100
train[nrow(train),10] = (sum(str_count(docs, ' mail '))/wordcount)*100
train[nrow(train),11] = (sum(str_count(docs, ' receive '))/wordcount)*100
train[nrow(train),12] = (sum(str_count(docs, ' will '))/wordcount)*100
train[nrow(train),13] = (sum(str_count(docs, ' people '))/wordcount)*100
train[nrow(train),14] = (sum(str_count(docs, ' report '))/wordcount)*100
train[nrow(train),15] = (sum(str_count(docs, ' addresses '))/wordcount)*100
train[nrow(train),16] = (sum(str_count(docs, ' free '))/wordcount)*100
train[nrow(train),17] = (sum(str_count(docs, ' business '))/wordcount)*100
train[nrow(train),18] = (sum(str_count(docs, ' email '))/wordcount)*100
train[nrow(train),19] = (sum(str_count(docs, ' you '))/wordcount)*100
train[nrow(train),20] = (sum(str_count(docs, ' credit '))/wordcount)*100
train[nrow(train),21] = (sum(str_count(docs, ' your '))/wordcount)*100
train[nrow(train),22] = (sum(str_count(docs, ' font '))/wordcount)*100
train[nrow(train),23] = (sum(str_count(docs, ' 000 '))/wordcount)*100
train[nrow(train),24] = (sum(str_count(docs, ' money '))/wordcount)*100
train[nrow(train),25] = (sum(str_count(docs, ' hp '))/wordcount)*100
train[nrow(train),26] = (sum(str_count(docs, ' hpl '))/wordcount)*100
train[nrow(train),27] = (sum(str_count(docs, ' george '))/wordcount)*100
train[nrow(train),28] = (sum(str_count(docs, ' 650 '))/wordcount)*100
train[nrow(train),29] = (sum(str_count(docs, ' lab '))/wordcount)*100
train[nrow(train),30] = (sum(str_count(docs, ' labs '))/wordcount)*100
train[nrow(train),31] = (sum(str_count(docs, ' telnet '))/wordcount)*100
train[nrow(train),32] = (sum(str_count(docs, ' 857 '))/wordcount)*100
train[nrow(train),33] = (sum(str_count(docs, ' data '))/wordcount)*100
train[nrow(train),34] = (sum(str_count(docs, ' 415 '))/wordcount)*100
train[nrow(train),35] = (sum(str_count(docs, ' 85 '))/wordcount)*100
train[nrow(train),36] = (sum(str_count(docs, ' technology '))/wordcount)*100
train[nrow(train),37] = (sum(str_count(docs, ' 1999 '))/wordcount)*100
train[nrow(train),38] = (sum(str_count(docs, ' parts '))/wordcount)*100
train[nrow(train),39] = (sum(str_count(docs, ' pm '))/wordcount)*100
train[nrow(train),40] = (sum(str_count(docs, ' direct '))/wordcount)*100
train[nrow(train),41] = (sum(str_count(docs, ' cs '))/wordcount)*100
train[nrow(train),42] = (sum(str_count(docs, ' meeting '))/wordcount)*100
train[nrow(train),43] = (sum(str_count(docs, ' original '))/wordcount)*100
train[nrow(train),44] = (sum(str_count(docs, ' project '))/wordcount)*100
train[nrow(train),45] = (sum(str_count(docs, ' re '))/wordcount)*100
train[nrow(train),46] = (sum(str_count(docs, ' edu '))/wordcount)*100
train[nrow(train),47] = (sum(str_count(docs, ' table '))/wordcount)*100
train[nrow(train),48] = (sum(str_count(docs, ' conference '))/wordcount)*100
```

``` r
#compute various attributes of capital run lengths
a= str_count(crude, '[A-Z]')
cap_total = sum(sapply(regmatches(crude, gregexpr("[A-Z]", crude, perl=TRUE)), length))
cap_number = sum(sapply(regmatches(crude, gregexpr("[A-Z]+", crude, perl=TRUE)), length))
cap_average = cap_total/cap_number
cap_longest = max(sapply(regmatches(crude, gregexpr("[A-Z]", crude, perl=TRUE)), length))

train[nrow(train),55] = cap_average
train[nrow(train),56] = cap_longest
train[nrow(train),57] = cap_total

#Match with different characters
total_char = sum(str_count(crude, '[a-z-A-Z-\\;\\(\\[\\!\\$\\#\\,\\.\\:]'))
total_char
```

    ## [1] 434

``` r
#Add the percentage of different characters in the data frame
train[nrow(train),49] = (sum(str_count(crude, '\\;'))/total_char)*100
train[nrow(train),50] = (sum(str_count(crude, '\\('))/total_char)*100
train[nrow(train),51] = (sum(str_count(crude, '\\['))/total_char)*100
train[nrow(train),52] = (sum(str_count(crude, '\\!'))/total_char)*100
train[nrow(train),53] = (sum(str_count(crude, '\\$'))/total_char)*100
train[nrow(train),54] = (sum(str_count(crude, '\\#'))/total_char)*100
```

``` r
#Classify using KNN algorithm and print output
test = train[nrow(train),]
trainx = train[-(nrow(train)),]
classifier1 = knn(trainx[-c(58)], test[-c(58)], trainx$spam, k = 5, algorithm=c("kd_tree"))

if (classifier1 == 1){
  print ('The email is a spam')
}else{
  print('The email is not a spam')
}
```

    ## [1] "The email is a spam"

The email was correctly classified as spam. However, I would like to check what happens if I trust the K fold cross-validation and use K =1 instead of K=5

``` r
#Classify using KNN algorithm and print output
test = train[nrow(train),]
trainx = train[-(nrow(train)),]
classifier1 = knn(trainx[-c(58)], test[-c(58)], trainx$spam, k = 1, algorithm=c("kd_tree"))

if (classifier1 == 1){
  print ('The email is a spam')
}else{
  print('The email is not a spam')
}
```

    ## [1] "The email is not a spam"

The spam email is incorrectly classfied as non-spam
