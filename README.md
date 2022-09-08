# Bert4Rec
This project protoypes a Bert-based sequential recommender engine and provide training/evaluation on internal dataset to predict next movies preference from user sessions


The prototype fully relies on the model from paper [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690v2). This paper improves existing collaborative filtering based engine by using user past clicks from a session to predict next click.

At a glance, it proposes a Bert architecture to be pretrained (with masked language modelling only) on all sessions (sequence of movies indices).
Then the model is used in production by right-appending a mask token to input session, so that
 the forwarding fill the mask with the most probable movie.
 
It gets advantages of bi-directional objective but due to nature of inputs you cannot benefit from large corpus pretraining like we tradiionally do with Bert on raw texts.
 
The notebook `Bert4Rec.ipynb` is standalone and runnable on colab. It is your responsability to provide path to the `recommandation.zip` TFrecord containing input movies sessions (the file is to big to be uploaded and possibly subject to restrictions...)
 
This notebook can be summarized as follow:
 
*  a `Data` section that read TFRecord to construct a `tf.Data.Dataset`. On particular, it prepares train/val/test sessions (padding, mask token) to feed model.
 
*  a `Modelling` section to instantiates the model. Here we take advantage of `transformers` and `datasets` packages

*  an `Evaluation` section to provide Hit Max Ratio {k} metric (very similar to top-k accuracy in classifiation)

As future work:

*  pretraining mask ratio should be grid searched using validation set

*  The training should not only mask a given session randomly but also force last session element masking.

*  Implementation can benefit from higher level interface using specific package [transformers4rec](https://pypi.org/project/transformers4rec/).

*  Use the movies titles from to propose a language-based movie embedding instead of one hot encoded movie indices.
