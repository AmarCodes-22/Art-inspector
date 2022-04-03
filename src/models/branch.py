from torch import nn


class Branch(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, hidden_dims: int = 256
    ) -> None:
        """Initialize a MLP for the task specific head

        Parameters
        ----------
        in_features : int
            Input features to the MLP, should be same as out_features from Stem
        out_features : int
            Out features to the MLP, should be equal to the number of classes for that
            task
        hidden_dims : int, optional
            Number of units in the hidden layer of MLP, by default 256
        """
        super(Branch, self).__init__()
        self.input_size = in_features
        self.linear1 = nn.Linear(in_features, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
