from typing import List
import torch
from torch import nn

from loguru import logger

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def scale(data, min_rating: int = 1, max_rating: int = 5):
    return (data - min_rating) / (max_rating - min_rating)

def unscale(data, min_rating: int = 1, max_rating: int = 5):
    return (data * (max_rating - min_rating)) + min_rating


class MatrixFactorization(nn.Module):
    """Matrix factorization model."""

    def __init__(
            self,
             u_features: torch.Tensor,
             v_features: torch.Tensor,
             ):
        super().__init__()
        self.u_features = u_features
        self.v_features = v_features

    def forward(
            self,
            ):        
        return torch.sigmoid(
            torch.matmul(self.u_features, self.v_features.t())
        )
    

class Loss(nn.Module):
    """docstring"""
    def __init__(
            self,
            # matrix: torch.Tensor,
            # non_zero_mask: torch.Tensor = None,
            lam_u: float = 0.3,
            lam_v: float = 0.3,
    ):
        super().__init__()
        self.lam_u = lam_u
        self.lam_v = lam_v

    def forward(
            self,
            matrix: torch.Tensor,
            non_zero_mask: torch.Tensor,
            predicted: torch.Tensor,
             u_features: torch.Tensor,
             v_features: torch.Tensor,
            ):
        diff = (matrix - predicted)**2
        prediction_error = torch.nansum(diff*non_zero_mask)

        # not nansum because no NaN values are expected
        u_regularization = self.lam_u * torch.sum(u_features.norm(dim=1))
        v_regularization = self.lam_v * torch.sum(v_features.norm(dim=1))
        
        return prediction_error + u_regularization + v_regularization
    


class ModelTrainer(nn.Module):
    """Wrapper class for the model class, loss class and optimizer class"""
    def __init__(
            self,
            n_features: int,
            loss_class: nn.Module, # a class instance
            model_class_type: nn.Module = MatrixFactorization, # a class type
            optimizer_class_type = torch.optim.Adam, # a subclass of torch.optim.Optimizer, not a class instance
    ):
        super().__init__()
        self.n_features = n_features
        self.loss_class = loss_class
        self._model_class_type = model_class_type
        self._optimizer_class_type = optimizer_class_type
        # Allows training to restart
        self.total_n_epochs: int = 0

    def train(self, matrix: torch.Tensor, n_epochs: int,
              u_features: torch.Tensor = None, v_features: torch.Tensor = None,
               non_zero_mask: torch.Tensor = None, lr: float=0.01
              ) -> None:
        """Set up variables and train a model."""
        # Scale the data if necessary. Save the min and max values for unscaling
        # NaNs first have to be replaced so that the min/max can be
        # calculated properly
        min_value = matrix.nan_to_num(nan=torch.inf).min()
        max_value = matrix.nan_to_num(nan=-torch.inf).max()
        if (min_value != 0) or (max_value != 1):
            # Data is not 0-1 scaled
            logger.debug("Scaling training data.")
            matrix = scale(matrix, min_rating=min_value, max_rating=max_value)
            self.min_matrix_value = min_value
            self.max_matrix_value = max_value
        else:
            self.min_matrix_value = None
            self.max_matrix_value = None
        
        # Set matrix and mask
        if non_zero_mask:
            assert matrix.shape == non_zero_mask.shape
        else:
            non_zero_mask = (~matrix.isnan())

        # Replace missings with -1 before assigning it to the attribute
        self.matrix = matrix.nan_to_num(nan=-1)
        self.non_zero_mask = non_zero_mask

        # Set feature vectors
        n_users, n_movies = matrix.shape
        if not u_features:
            u_features = torch.randn(
                n_users, self.n_features, requires_grad=True, device=DEVICE
                )
        else:
            assert u_features.shape == (n_users, self.n_features)

        if not v_features:
            v_features = torch.randn(
                n_movies, self.n_features, requires_grad=True, device=DEVICE
                )
        else:
            assert v_features.shape == (n_users, self.n_features)

        self.u_features = u_features
        self.v_features = v_features

        # Set model and optimiser
        self.model = self._model_class_type(self.u_features, self.v_features)
        self.optimizer = self._optimizer_class_type(
            [self.u_features, self.v_features], lr=lr)
        
        epochs_so_far = self.total_n_epochs
        target_n_epochs = epochs_so_far+n_epochs
        for i in range(epochs_so_far, target_n_epochs):
            self._train()
            # Increase epoch count after a successful training
            self.total_n_epochs += 1
            if i % 10 == 0:
                print(f"Epoch: {i}/{n_epochs}")
                self._validate()
        return None

    def _train(self):
            """Perform one training loop"""
            self.optimizer.zero_grad()

            predicted = self.model()
            loss = self.loss_class(
                matrix= self.matrix,
                non_zero_mask= self.non_zero_mask,
                predicted= predicted,
                u_features = self.u_features,
                v_features = self.v_features,
            )
            loss.backward()
            self.optimizer.step()

    def _validate(self):
        """Validate data during training."""
        # here is no validation dataset here so we just made predictions of the whole dataset
        predicted, actual = self.predict()
        # Convert bool to float and find the mean
        score = (predicted == actual).float().nanmean()
        print(f"Current score: {score}")
    
    def predict(self, user_idx: List[int] = None, matrix: torch.Tensor = None):
        """Predict movie ratings for given users.
        
        Data are scaled and unscaled if necessary.
        """
        # Scale if necessary
        scaling_info_available = (
            (self.min_matrix_value is not None) and
            (self.max_matrix_value is not None)
        )
        if matrix:
            matrix_min = matrix.nan_to_num(nan=torch.inf).min()
            matrix_max = matrix.nan_to_num(nan=-torch.inf).max()
            if not ((matrix_min >= 0) and (matrix_max <= 1)):
                # data is not scaled. We need to scale the data.
                if not scaling_info_available:
                    raise ValueError(
                        "Data need to be scaled but scaling properties haven't be set."
                    )
                logger.debug("Scaling data for predicting.")
                matrix = scale(matrix,
                               min_rating=self.min_matrix_value,
                               max_rating=self.max_matrix_value)
            matrix = matrix.nan_to_num(nan=-1)
        else:
            # self.matrix was already scaled in self.train()
            # Nans were also replaced with -1
            matrix = self.matrix

        # Set user_idx
        if user_idx is None:
            user_idx = torch.arange(0,matrix.size(0), dtype=torch.int)
        if isinstance(user_idx, int):
            user_idx = [user_idx]
        
        # Get predictions
        predicted_ratings, actual_ratings, non_zero_mask =  (
            self._predict(matrix, user_idx)
        )

        # Convert the -1 back into NaN. We don't do this for 
        # the predictionss
        actual_ratings[~non_zero_mask] = torch.nan

        # Unscale if necessary
        if scaling_info_available:
            logger.debug("Unscaling predicted results.")
            predicted_ratings = unscale(
                predicted_ratings,
                min_rating=self.min_matrix_value, max_rating=self.max_matrix_value
                )
            actual_ratings = unscale(
                actual_ratings,
            min_rating=self.min_matrix_value, max_rating=self.max_matrix_value

                )
        return predicted_ratings.round(), actual_ratings.round()
    
    def _predict(self, matrix: torch.Tensor, user_idx: List[int]):
        """Make prediction on scaled data.
        """
        user_ratings = matrix[user_idx, :]
        non_zero_mask = user_ratings != -1

        with torch.no_grad():
            predictions = torch.sigmoid(
                torch.mm(
                    self.model.u_features[user_idx, :].view(-1, self.n_features),
                    self.model.v_features.t())
                )
            
        # Also returning the y_pred where there was no y_true
        # predicted_ratings = predictions.squeeze()[non_zero_mask]
        # actual_ratings = user_ratings[non_zero_mask]
        # NOTE: These values are not scaled
        return predictions, user_ratings, non_zero_mask


        

