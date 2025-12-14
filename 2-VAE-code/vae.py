import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os, json

# In TensorFlow 2.17 the functional alias keras.losses.mean_squared_error(y_true, y_pred) has been removed. Use mse instead.

class VAE_bilayer(keras.Model):
    """
    Two‐layer Variational Autoencoder (VAE) in Keras.
    
    Args:
        original_dim (int): Dimension of the input vector.
        hidden_dim1  (int): Units in the first hidden layer.
        hidden_dim2  (int): Units in the second hidden layer.
        latent_dim   (int): Dimensionality of the latent code z.
        loss_type    (str): One of
                           - "mean_squared_error" (MSE only)
                           - "cosine_similarity"  (cos only)
                           - "RMSE+KL"            (MSE + KL)
                           - "cos+KL"             (cos + KL)
        rec_weight   (float): Weighting factor for the reconstruction term when combined with KL.
        dropout_rate (float): Dropout fraction in the encoder.
    """
    def __init__(self,
                 original_dim: int,
                 hidden_dim1:  int,
                 hidden_dim2:  int,
                 latent_dim:   int,
                 loss_type:    str = "RMSE+KL",
                 rec_weight:   float = 1.0,
                 dropout_rate: float = 0.25,
                 **kwargs):
        super().__init__(**kwargs)
        # Store config
        self.original_dim = original_dim
        self.loss_type    = loss_type
        self.rec_weight   = rec_weight
        self.dropout_rate = dropout_rate
        
        # ===== Encoder layers =====
        self.dense_e1   = layers.Dense(hidden_dim1, activation="relu", name="enc_dense1")
        self.dropout    = layers.Dropout(dropout_rate, name="enc_dropout")
        self.dense_e2   = layers.Dense(hidden_dim2, activation="relu", name="enc_dense2")
        self.dense_mu   = layers.Dense(latent_dim,  name="enc_mu")
        self.dense_logv = layers.Dense(latent_dim,  name="enc_logvar")
        
        # ===== Decoder layers =====
        self.dense_d1   = layers.Dense(hidden_dim2, activation="relu", name="dec_dense1")
        self.dense_d2   = layers.Dense(hidden_dim1, activation="relu", name="dec_dense2")
        self.output_dec = layers.Dense(original_dim, activation="sigmoid", name="dec_output")
        
        # ===== Loss & metric trackers =====
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.rec_loss_tracker   = keras.metrics.Mean(name="loss_reconstruction")
        self.kl_loss_tracker    = keras.metrics.Mean(name="loss_kl")
    
    @property
    def metrics(self):
        # Keras will reset these trackers at the start of each epoch
        return [self.total_loss_tracker,
                self.rec_loss_tracker,
                self.kl_loss_tracker]
    
    def encode(self, x):
        h = self.dense_e1(x)
        h = self.dropout(h)
        h = self.dense_e2(h)
        mu     = self.dense_mu(h)
        logvar = self.dense_logv(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps
    
    def decode(self, z):
        h = self.dense_d1(z)
        h = self.dense_d2(h)
        return self.output_dec(h)
    
    def call(self, inputs, training=False):
        # For inference: returns only the reconstruction
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)
    
    def train_step(self, data):
        # Unpack data
        x = data if isinstance(data, tf.Tensor) else data[0]
        
        with tf.GradientTape() as tape:
            # Forward pass
            mu, logvar    = self.encode(x)
            z             = self.reparameterize(mu, logvar)
            reconstruction= self.decode(z)
            
            # Reconstruction loss
            if self.loss_type in ("mean_squared_error", "RMSE+KL"):
                rec_loss = tf.reduce_mean(
                    keras.losses.mse(x, reconstruction)
                ) * self.rec_weight * self.original_dim
            elif self.loss_type in ("cosine_similarity", "cos+KL"):
                # cosine_similarity yields –1 when identical; shift to positive
                rec_loss = tf.reduce_mean(
                    1 + keras.losses.cosine_similarity(x, reconstruction)
                ) * self.rec_weight * self.original_dim
            else:
                # any other built‐in Keras loss
                base = keras.losses.get(self.loss_type)
                rec_loss = tf.reduce_mean(base(x, reconstruction))
            
            # KL divergence term (per batch)
            kl_loss = -0.5 * tf.reduce_mean(
                1 + logvar - tf.square(mu) - tf.exp(logvar)
            )
            
            # Total ELBO
            total_loss = rec_loss + kl_loss if "KL" in self.loss_type else rec_loss
        
        # Backpropagate
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Update trackers
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss":                self.total_loss_tracker.result(),
            "loss_reconstruction": self.rec_loss_tracker.result(),
            "loss_kl":             self.kl_loss_tracker.result()
        }
    
    def test_step(self, data):
        # Same as train_step but without gradients
        x = data if isinstance(data, tf.Tensor) else data[0]
        mu, logvar     = self.encode(x)
        z              = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        if self.loss_type in ("mean_squared_error", "RMSE+KL"):
            rec_loss = tf.reduce_mean(
                keras.losses.mse(x, reconstruction)
            ) * self.rec_weight * self.original_dim
        elif self.loss_type in ("cosine_similarity", "cos+KL"):
            rec_loss = tf.reduce_mean(
                1 + keras.losses.cosine_similarity(x, reconstruction)
            ) * self.rec_weight * self.original_dim
        else:
            base     = keras.losses.get(self.loss_type)
            rec_loss = tf.reduce_mean(base(x, reconstruction))
        
        kl_loss    = -0.5 * tf.reduce_mean(
                        1 + logvar - tf.square(mu) - tf.exp(logvar)
                     )
        total_loss = rec_loss + kl_loss if "KL" in self.loss_type else rec_loss
        # total_loss = K.mean(0.9 * (rec_loss) + 0.1 * (kl_loss))
        
        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss":                self.total_loss_tracker.result(),
            "loss_reconstruction": self.rec_loss_tracker.result(),
            "loss_kl":             self.kl_loss_tracker.result()
        }

    def get_config(self):
        """
        Return the arguments needed to reconstruct the model.
        Keras will use this in serialization, but we also
        save it explicitly to JSON in our save method.
        """
        return {
            "original_dim": self.original_dim,
            "hidden_dim1":  self.dense_e1.units,
            "hidden_dim2":  self.dense_e2.units,
            "latent_dim":   self.dense_mu.units,
            "loss_type":    self.loss_type,
            "rec_weight":   self.rec_weight,
            "dropout_rate": self.dropout_rate,
        }

    def save(self, folder, 
    # save_weights_only: bool = False
    ):
        """
        Save both config and weights. If save_weights_only=True,
        only the weights are saved (still requires config.json to reload).
        """
        os.makedirs(folder, exist_ok=True)
        # 1) save config
        cfg = self.get_config()
        with open(os.path.join(folder, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        # 2) save weights
        # self.save_weights(os.path.join(folder, "vae.weights.h5"))
        # 3) optionally save full SavedModel
        # if not save_weights_only:
        #     tf.saved_model.save(self, os.path.join(folder, "saved_model"))

    @classmethod
    def load_vae(cls, folder, compile_kwargs=None):
        """
        Rebuild from saved config.json + weights.
        
        Args:
            folder: directory containing config.json and weights.h5
            compile_kwargs: dict passed to model.compile()
        """
        # 1) load config
        cfg_path = os.path.join(folder, "config.json")
        with open(cfg_path, "r") as f:
            config = json.load(f)
            print(config)
        
        # 2) reconstruct model
        model = cls(**config)
        
        # 3) compile with whatever optimizer/loss you like
        if compile_kwargs is None:
            compile_kwargs = {"optimizer": "adam"}
        model.compile(**compile_kwargs)
        model.build(input_shape=(None, config['original_dim']))
        
        # 4) load weights
        weights_path = os.path.join(folder, "vae.weights.h5")
        model.load_weights(weights_path)
        return model