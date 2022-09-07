# Bert4Rec
This project protoypes a model Bert based sequential recommender engine and provide train,/eval on internal dataset to predict next movies preference from user sessions


The prototype fully rely on the model from paper [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690v2). This paper improves existing collaborative filtering based engine by using user past clicks (session) to predict next click.

At a glance, it proposes a Bert architecture to be pretrained on all sessions (sequence of movies indices past perferences).
Then the model is used in production by right-appending a mask token to input session, so that
 the forwarding pass to guess the most probable movie.
 
It gets advantages of bi-directional objective but due to nature of inputs you cannot benefit from large corpus pretraining like we tradiionally do with Bert on raw texts.
 
The notebook `run.py` is standalone and runnable on colab. It is your responsability to provide path to the .zip TFrecord containing input movies sessions.
 
This notebook can be summarized as follow:
 
*  a `Data` section that read TFRecord to construct a `tf.Data.Dataset`. On particular, it prepares train/val/test sessions (padding, mask token) to feed model.
 
*  a `Modelling` section to instantiates the model. Here we take advantage of `transformers` and `datasets` packages

*  an `Evaluation` section to provide Hit Max Ratio {k} metric (very similar to top-k accuracy in classifiation)

As future work:

*  mask ratio should be grid searched using validation set

*  The training should not only mask a given session randomly but also force last session element masking.

*  Implementation can benefit from higher level interface using specific package [transformers4rec](https://pypi.org/project/transformers4rec/).
