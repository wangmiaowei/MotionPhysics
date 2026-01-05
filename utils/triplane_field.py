import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn
from dataclasses import dataclass
from typing import Union, Optional, Sequence, Tuple, List


@dataclass
class SceneBox:
    """Data to represent the scene box."""

    aabb: Float[Tensor, "2 3"]
    """aabb: axis-aligned bounding box.
    aabb[0] is the minimum (x,y,z) point.
    aabb[1] is the maximum (x,y,z) point."""

    def get_diagonal_length(self):
        """Returns the longest diagonal length."""
        diff = self.aabb[1] - self.aabb[0]
        length = torch.sqrt((diff**2).sum() + 1e-20)
        return length

    def get_center(self):
        """Returns the center of the box."""
        diff = self.aabb[1] - self.aabb[0]
        return self.aabb[0] + diff / 2.0

    def get_centered_and_scaled_scene_box(
        self, scale_factor: Union[float, torch.Tensor] = 1.0
    ):
        """Returns a new box that has been shifted and rescaled to be centered
        about the origin.

        Args:
            scale_factor: How much to scale the camera origins by.
        """
        return SceneBox(aabb=(self.aabb - self.get_center()) * scale_factor)

    @staticmethod
    def get_normalized_positions(
        positions: Float[Tensor, "*batch 3"], aabb: Float[Tensor, "2 3"]
    ):
        """Return normalized positions in range [0, 1] based on the aabb axis-aligned bounding box.

        Args:
            positions: the xyz positions
            aabb: the axis-aligned bounding box
        """
        aabb_lengths = aabb[1] - aabb[0]
        normalized_positions = (positions - aabb[0]) / aabb_lengths
        return normalized_positions

    @staticmethod
    def from_camera_poses(
        poses: Float[Tensor, "*batch 3 4"], scale_factor: float
    ) -> "SceneBox":
        """Returns the instance of SceneBox that fully envelopes a set of poses

        Args:
            poses: tensor of camera pose matrices
            scale_factor: How much to scale the camera origins by.
        """
        xyzs = poses[..., :3, -1]
        aabb = torch.stack([torch.min(xyzs, dim=0)[0], torch.max(xyzs, dim=0)[0]])
        return SceneBox(aabb=aabb * scale_factor)


def compute_plane_tv(t: torch.Tensor, only_w: bool = False) -> float:
    """Computes total variance across a plane.
    From nerf-studio

    Args:
        t: Plane tensor
        only_w: Whether to only compute total variance across w dimension

    Returns:
        Total variance
    """
    _, h, w = t.shape
    w_tv = torch.square(t[..., :, 1:] - t[..., :, : w - 1]).mean()

    if only_w:
        return w_tv

    h_tv = torch.square(t[..., 1:, :] - t[..., : h - 1, :]).mean()
    return h_tv + w_tv

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
        zero_init = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = (
            set(skip_connections) if skip_connections else set()
        )
        self.activation = activation
        self.out_activation = out_activation
        self.net = None
        self.zero_init = zero_init

        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """Initialize multi-layer perceptron."""
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.in_dim, self.out_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert (
                        i not in self._skip_connections
                    ), "Skip connection at layer 0 doesn't make sense."
                    layers.append(nn.Linear(self.in_dim, self.layer_width))
                elif i in self._skip_connections:
                    layers.append(
                        nn.Linear(self.layer_width + self.in_dim, self.layer_width)
                    )
                else:
                    layers.append(nn.Linear(self.layer_width, self.layer_width))
            layers.append(nn.Linear(self.layer_width, self.out_dim))
        self.layers = nn.ModuleList(layers)

        if self.zero_init:
            torch.nn.init.zeros_(self.layers[-1].weight)
            torch.nn.init.zeros_(self.layers[-1].bias)

    def pytorch_fwd(
        self, in_tensor: Float[Tensor, "*bs in_dim"]
    ) -> Float[Tensor, "*bs out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        x = in_tensor
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

    def forward(
        self, in_tensor: Float[Tensor, "*bs in_dim"]
    ) -> Float[Tensor, "*bs out_dim"]:
        return self.pytorch_fwd(in_tensor)


class TriplanesEncoding(nn.Module):
    """

    Args:
        resolutions (Sequence[int]): xyz resolutions.
    """

    def __init__(
        self,
        resolutions: Sequence[int],
        feat_dim: int = 32,
        init_a: float = 0.1,
        init_b: float = 0.5,
        reduce="sum",  # Literal["sum", "product", "cat"] = "sum",
    ):
        super().__init__()

        self.resolutions = resolutions

        if reduce == "cat":
            feat_dim = feat_dim  #  // 3
        self.feat_dim = feat_dim

        self.reduce = reduce

        self.in_dim = 3

        self.plane_coefs = nn.ParameterList()

        self.coo_combs = [[0, 1], [0, 2], [1, 2]]
        # [(x, t), (y, t), (z, t)]
        for coo_comb in self.coo_combs:
            new_plane_coef = nn.Parameter(
                torch.empty(
                    [
                        self.feat_dim,
                        resolutions[coo_comb[1]],
                        resolutions[coo_comb[0]],
                    ]
                )
            )

            # when init to ones?

            nn.init.uniform_(new_plane_coef, a=init_a, b=init_b)
            self.plane_coefs.append(new_plane_coef)

    def forward(self, inp: Float[Tensor, "*bs 3"]):
        output = 1.0 if self.reduce == "product" else 0.0
        if self.reduce == "cat":
            output = []
        for ci, coo_comb in enumerate(self.coo_combs):
            grid = self.plane_coefs[ci].unsqueeze(0)  # [1, feature_dim, reso1, reso2]
            coords = inp[..., coo_comb].view(1, 1, -1, 2)  # [1, 1, flattened_bs, 2]

            interp = F.grid_sample(
                grid, coords, align_corners=True, padding_mode="border"
            )  # [1, output_dim, 1, flattened_bs]
            interp = interp.view(self.feat_dim, -1).T  # [flattened_bs, output_dim]

            if self.reduce == "product":
                output = output * interp
            elif self.reduce == "sum":
                output = output + interp
            elif self.reduce == "cat":
                output.append(interp)

        if self.reduce == "cat":
            # [flattened_bs, output_dim * 3]
            output = torch.cat(output, dim=-1)

        return output

    def compute_plane_tv(
        self,
    ):
        ret_loss = 0.0

        for plane_coef in self.plane_coefs:
            ret_loss += compute_plane_tv(plane_coef)

        return ret_loss
    

class TriplaneFields(nn.Module):
    """Temporal Kplanes SE(3) fields.

    Args:
        aabb: axis-aligned bounding box.
            aabb[0] is the minimum (x,y,z) point.
            aabb[1] is the maximum (x,y,z) point.
        resolutions: resolutions of the kplanes. in an order of [x, y, z]

    """

    def __init__(
        self,
        aabb: Float[Tensor, "2 3"],
        resolutions: Sequence[int],
        feat_dim: int = 64,
        init_a: float = 0.1,
        init_b: float = 0.5,
        reduce="sum",  #: Literal["sum", "product", "cat"] = "sum",
        num_decoder_layers=2,
        decoder_hidden_size=64,
        output_dim: int = 96,
        zero_init: bool = False,
    ):
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.output_dim = output_dim

        self.kplanes_encoding = TriplanesEncoding(
            resolutions, feat_dim, init_a, init_b, reduce
        )

        if reduce == "cat":
            feat_dim = feat_dim * 3
        self.decoder = MLP(
            feat_dim,
            num_decoder_layers,
            layer_width=decoder_hidden_size,
            out_dim=self.output_dim,
            skip_connections=None,
            activation=nn.ReLU(),
            out_activation=None,
            zero_init=zero_init,
        )

    def forward(
        self, inp: Float[Tensor, "*bs 3"]
    ) -> Tuple[Float[Tensor, "*bs 3 3"], Float[Tensor, "*bs 3"]]:
        # shift to [-1, 1]
        inpx = SceneBox.get_normalized_positions(inp, self.aabb) * 2.0 - 1.0

        output = self.kplanes_encoding(inpx)

        output = self.decoder(output)

        # split_size = output.shape[-1] // 3
        # output = torch.stack(torch.split(output, split_size, dim=-1), dim=-1)

        return output

    def compute_smoothess_loss(
        self,
    ):
        smothness_loss = self.kplanes_encoding.compute_plane_tv()

        return smothness_loss


def compute_entropy(p):
    return -torch.sum(
        p * torch.log(p + 1e-5), dim=1
    ).mean()  # Adding a small constant to prevent log(0)


class TriplaneFieldsWithEntropy(nn.Module):
    """Temporal Kplanes SE(3) fields.

    Args:
        aabb: axis-aligned bounding box.
            aabb[0] is the minimum (x,y,z) point.
            aabb[1] is the maximum (x,y,z) point.
        resolutions: resolutions of the kplanes. in an order of [x, y, z]

    """

    def __init__(
        self,
        aabb: Float[Tensor, "2 3"],
        resolutions: Sequence[int],
        feat_dim: int = 64,
        init_a: float = 0.1,
        init_b: float = 0.5,
        reduce="sum",  #: Literal["sum", "product", "cat"] = "sum",
        num_decoder_layers=2,
        decoder_hidden_size=64,
        output_dim: int = 96,
        zero_init: bool = False,
        num_cls: int = 3,
    ):
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.output_dim = output_dim
        self.num_cls = num_cls

        self.kplanes_encoding = TriplanesEncoding(
            resolutions, feat_dim, init_a, init_b, reduce
        )

        self.decoder = MLP(
            feat_dim,
            num_decoder_layers,
            layer_width=decoder_hidden_size,
            out_dim=self.num_cls,
            skip_connections=None,
            activation=nn.ReLU(),
            out_activation=None,
            zero_init=zero_init,
        )

        self.cls_embedding = torch.nn.Embedding(num_cls, output_dim)

    def forward(
        self, inp: Float[Tensor, "*bs 3"]
    ) -> Tuple[Float[Tensor, "*bs 3 3"], Float[Tensor, "1"]]:
        # shift to [-1, 1]
        inpx = SceneBox.get_normalized_positions(inp, self.aabb) * 2.0 - 1.0

        output = self.kplanes_encoding(inpx)

        output = self.decoder(output)

        prob = F.softmax(output, dim=-1)

        entropy = compute_entropy(prob)

        cls_index = torch.tensor([0, 1, 2]).to(inp.device)
        cls_emb = self.cls_embedding(cls_index)

        output = torch.matmul(prob, cls_emb)

        return output, entropy

    def compute_smoothess_loss(
        self,
    ):
        smothness_loss = self.kplanes_encoding.compute_plane_tv()

        return smothness_loss
