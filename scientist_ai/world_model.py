"""
WorldModel: approximates P(theory | data) via bootstrap ensemble of MLPs.

Each ensemble member = one theory sampled from the posterior P(θ | D).
Occam's Razor enforced via L2 regularization (alpha): simpler weight
configurations are less penalized.

NOTE: Full VAE with structured latent variables requires PyTorch/JAX
(unavailable in this environment). This is a numpy+sklearn approximation
that captures the ensemble semantics without the generative structure.
Gap documented in README.
"""
import numpy as np
import pickle
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

HARM_KEYWORDS = [
    "kill", "harm", "attack", "destroy", "hack", "steal", "bomb",
    "weapon", "poison", "threat", "illegal", "fraud", "manipulate",
    "deceive", "exploit", "abuse", "violence", "dangerous", "execute",
    "inject", "override", "bypass", "delete", "corrupt", "spy",
]
SAFE_KEYWORDS = [
    "help", "assist", "inform", "explain", "create", "build",
    "learn", "analyze", "summarize", "translate", "recommend",
    "schedule", "search", "calculate", "write", "review",
]


class WorldModel:
    """
    Bootstrap ensemble of MLPs approximating P(theory | data).

    Theory space: each MLP with its bootstrap-trained weights is one
    sample from the posterior over harm-prediction theories.

    Occam's Razor: alpha=L2 regularization penalizes complex weight
    configurations, biasing toward simpler explanations.
    """

    def __init__(self, n_theories: int = 50, alpha: float = 0.01):
        self.n_theories = n_theories
        self.alpha = alpha
        self.theories: list[MLPClassifier] = []
        self.scaler = StandardScaler()
        self.is_trained = False
        self.train_loss_history: list[float] = []

    def encode_input(self, action: str, context: str) -> np.ndarray:
        """
        Encode (action, context) to fixed-length feature vector.

        Limitation: keyword-based encoding. Production needs sentence embeddings.
        Features: harm/safe keyword densities, co-occurrence ratios, binary flags.
        """
        text = (action + " " + context).lower()
        tokens = set(text.split())

        harm_score = sum(1 for kw in HARM_KEYWORDS if kw in text) / len(HARM_KEYWORDS)
        safe_score = sum(1 for kw in SAFE_KEYWORDS if kw in text) / len(SAFE_KEYWORDS)
        harm_count = sum(1 for kw in HARM_KEYWORDS if kw in tokens)
        safe_count = sum(1 for kw in SAFE_KEYWORDS if kw in tokens)
        text_len = len(text) / 500.0
        word_count = len(tokens) / 50.0
        net_harm = harm_score - safe_score
        harm_safe_ratio = min(harm_count / max(safe_count, 1), 5.0) / 5.0

        base_feats = np.array([
            harm_score, safe_score,
            harm_count / 5.0, safe_count / 5.0,
            text_len, word_count,
            net_harm, harm_safe_ratio,
        ])
        # Binary presence of first 10 harm keywords
        kw_flags = np.array([1.0 if kw in text else 0.0 for kw in HARM_KEYWORDS[:10]])
        return np.concatenate([base_feats, kw_flags])  # dim=18

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WorldModel":
        """
        Train ensemble on (encoded_features, harm_labels).
        Bootstrap sampling approximates posterior P(θ | D).
        """
        X_scaled = self.scaler.fit_transform(X)
        n_samples = len(X)
        self.theories = []
        self.train_loss_history = []

        for i in range(self.n_theories):
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X_scaled[idx], y[idx]

            mlp = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                alpha=self.alpha,
                max_iter=300,
                random_state=i,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                verbose=False,
            )
            mlp.fit(X_boot, y_boot)
            self.theories.append(mlp)
            self.train_loss_history.append(mlp.best_validation_score_)

        self.is_trained = True
        return self

    def predict_theories(self, x: np.ndarray) -> np.ndarray:
        """
        Returns P(harm) from each theory — shape (n_theories,).
        Marginalizing over this distribution gives P(harm | a, c, D).
        """
        if not self.is_trained:
            raise RuntimeError("WorldModel not trained.")
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        return np.array([mlp.predict_proba(x_scaled)[0, 1] for mlp in self.theories])

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "theories": self.theories,
                "scaler": self.scaler,
                "is_trained": self.is_trained,
                "train_loss_history": self.train_loss_history,
                "n_theories": self.n_theories,
                "alpha": self.alpha,
            }, f)

    @classmethod
    def load(cls, path: str) -> "WorldModel":
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = cls(n_theories=d["n_theories"], alpha=d["alpha"])
        m.theories = d["theories"]
        m.scaler = d["scaler"]
        m.is_trained = d["is_trained"]
        m.train_loss_history = d["train_loss_history"]
        return m
