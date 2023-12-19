# Multichannel CNN for Toxic Comment Classification

This project addresses the pervasive issue of toxic commentary online, which poses a significant threat to the integrity of digital discourse. Toxic comments can create environments of insecurity, obstruct productive exchanges, and lead to severe psychological effects on the recipients. The objective of this project is to detect and categorize such toxic comments, thereby contributing to the cultivation of a secure and respectful online community.

## Project Overview

The multifaceted nature of toxic comments, which includes a spectrum of derogatory, insulting, or outright harassing language, requires a nuanced approach for detection and classification. This project employs a multichannel Convolutional Neural Network (CNN) architecture, which is particularly adept at identifying complex patterns within text data. By processing various representations of text input in parallel, the multichannel CNN enhances the model's sensitivity to the intricate features of language used in toxic comments. 
The model has demonstrated excellent performance, achieving an **accuracy of 95% on the training data and 97% on the validation data.**

## Dataset

The dataset leveraged for this project is derived from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) on Kaggle, featuring a comprehensive set of comments from Wikipedia discussions tagged for various degrees of toxicity.

## Word Embeddings

For textual analysis, the project utilizes 100-dimensional GloVe word embeddings, which provide a dense representation of words and their contextual meanings. These embeddings play a pivotal role in enabling the neural network to interpret and evaluate the textual data effectively. Access the GloVe embeddings from the table below.

## How to Use

Each component of the project plays a critical role in the overall functionality:

- **Streamlit App**: This is the web application interface where users can interact with the model for making predictions.
- **Model**: The trained multichannel CNN model file which is used for prediction.
- **Tokenizer Pickle**: This contains the tokenization mapping used by the model to convert text data into a format that can be processed.

(Include additional instructions on how to use the Streamlit app, how to load the model, and how to apply the tokenizer for new data.)

## Useful Links Table

| Description        | Link                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------- |
| Dataset            | [Kaggle Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)  |
| Word Embeddings    | [GloVe 100D](https://drive.google.com/file/d/1cA5jYv_n9rIrBuhqc1HfnnfCQlojKKXV/view?usp=sharing) |
| Streamlit App      | [Streamlit Interface](https://drive.google.com/file/d/1pApTXQLimki7jOTQkRb2Ht4tyVtglZWF/view?usp=sharing) |
| Model              | [CNN Model](https://drive.google.com/file/d/1iTxhqpA8rHhCelQxu9DAN20NzNwd49nj/view?usp=sharing) |
| Tokenizer Pickle   | [Tokenizer](https://drive.google.com/file/d/18DN21jgVVaEdov5liEG_ez49s8PoxyaN/view?usp=sharing) |
| Streamlit Deploy   | (Insert the Streamlit deployment link here)                                                |
| YoutubeComment     | You Can Dowload in File Above                                                              |

## Contributors

1. Robby Hidayah Ramadhan - 120450033
2. Muhammad Aqsal Fadillah - 120450077
3. Nadhira Adela Putri - 12045001
4. Wulan Ayu Windari - 120450045

## License
Deep Learning Project [RA] - 2023

