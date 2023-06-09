import torch
import cv2, os
import numpy as np
import math, copy
from utils import one_hot_encode, output_onehot2text, lengths_to_mask, turn_text2image, load_data, add_recon_title
from torchvision.utils import make_grid
import imageio
import ssl
import torchvision

class BaseDataset():
    """
    Abstract dataset class shared for all datasets
    """

    def __init__(self, pth, testpth, mod_type):
        """

        :param pth: path to the given modality
        :type pth: str
        :param mod_type: tag for the modality for correct processing (e.g. "text", "image", "mnist", "svhn" etc.)
        :type mod_type: str
        """
        assert hasattr(self, "feature_dims"), "Dataset class must have the feature_dims attribute"
        self.path = pth
        self.testdata = testpth
        self.current_path = None
        self.mod_type = mod_type
        self.has_masks = False
        self.categorical = False

    def _mod_specific_loaders(self):
        """
        Assigns the preprocessing function based on the mod_type
        """
        raise NotImplementedError

    def _mod_specific_savers(self):
        """
        Assigns the postprocessing function based on the mod_type
        """
        raise NotImplementedError

    def labels(self):
        """Returns labels for the whole dataset"""
        return None

    def get_labels(self, split="train"):
        """Returns labels for the given split: train or test"""
        self.current_path = self.path if split == "train" else self.testdata
        return self.labels()

    def eval_statistics_fn(self):
        """(optional) Returns a dataset-specific function that runs systematic evaluation"""
        return None

    def current_datatype(self):
        """Returns whther the current path to data points to test data or train data"""
        if self.current_path == self.testdata:
            return "test"
        elif self.current_path == self.path:
            return "train"
        else:
            return None

    def _preprocess(self):
        """
        Preprocesses the loaded data according to modality type

        :return: preprocessed data
        :rtype: list
        """
        assert self.mod_type in self._mod_specific_loaders().keys(), "Unsupported modality type for {}".format(
            self.current_path)
        return self._mod_specific_loaders()[self.mod_type]()

    def _postprocess(self, output_data):
        """
        Postprocesses the output data according to modality type

        :return: postprocessed data
        :rtype: list
        """
        assert self.mod_type in self._mod_specific_savers().keys(), "Unsupported modality type for {}".format(self.current_path)
        return self._mod_specific_savers()[self.mod_type](output_data)

    def get_processed_recons(self, recons_raw):
        """
        Returns the postprocessed data that came from the decoders

        :param recons_raw: tensor with output reconstructions
        :type recons_raw: torch.tensor
        :return: postprocessed data as returned by the specific _postprocess function
        :rtype: list
        """
        return self._postprocess(recons_raw)

    def get_data_raw(self):
        """
        Loads raw data from path

        :return: loaded raw data
        :rtype: list
        """
        data = load_data(self.current_path)
        return data

    def get_data(self):
        """
        Returns processed data

        :return: processed data
        :rtype: list
        """
        self.current_path = self.path
        return self._preprocess()

    def get_test_data(self):
        """
        Returns processed test data if available

        :return: processed data
        :rtype: list
        """
        if self.testdata is not None:
            self.current_path = self.testdata
            return self._preprocess()
        return None

    def _preprocess_images(self, dimensions):
        """
        General function for loading images and preparing them as torch tensors

        :param dimensions: feature_dim for the image modality
        :type dimensions: list
        :return: preprocessed data
        :rtype: torch.tensor
        """
        data = [torch.from_numpy(np.asarray(x.reshape(*dimensions)).astype(np.float)) for x in self.get_data_raw()]
        return torch.stack(data)

    def _preprocess_text_onehot(self):
        """
        General function for loading text strings and preparing them as torch one-hot encodings

        :return: torch with text encodings and masks
        :rtype: torch.tensor
        """
        self.has_masks = True
        self.categorical = True
        data = []
        for x in self.get_data_raw():
            d = " ".join(x) if isinstance(x, list) else x
            data.append(d)
        data = [one_hot_encode(len(f), f) for f in data]
        data = [torch.from_numpy(np.asarray(x)) for x in data]
        masks = lengths_to_mask(torch.tensor(np.asarray([x.shape[0] for x in data]))).unsqueeze(-1)
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0.0)
        data_and_masks = torch.cat((data, masks), dim=-1)
        return data_and_masks

    def _postprocess_all2img(self, data):
        """
        Converts any kind of data to images to save traversal visualizations

        :param data: input data
        :type data: torch.tensor
        :return: processed images
        :rtype: torch.tensor
        """
        output_processed = self._postprocess(data)
        output_processed = turn_text2image(output_processed, img_size=self.text2img_size) \
            if self.mod_type == "text" else output_processed
        return output_processed

    def save_traversals(self, recons, path, num_dims):
        """
        Makes a grid of traversals and saves as image

        :param recons: data to save
        :type recons: torch.tensor
        :param path: path to save the traversal to
        :type path: str
        :param num_dims: number of latent dimensions
        :type num_dims: int
        """
        if len(recons.shape) < 3:
            output_processed = torch.tensor(np.asarray(self._postprocess_all2img(recons))).transpose(1, 3)
            grid = np.asarray(make_grid(output_processed, padding=1, nrow=int(math.sqrt(len(recons)))).transpose(2, 0))
            cv2.imwrite(path, cv2.cvtColor(grid.astype("uint8"), cv2.COLOR_BGR2RGB))
        else:
            output_processed = torch.stack([torch.tensor(self._postprocess_all2img(x)) for x in recons])
            output_processed = output_processed.reshape(num_dims, -1, *output_processed.shape[1:]).squeeze()
            rows = []
            for ind, dim in enumerate(output_processed):
                rows.append(np.asarray(torch.hstack([x for x in dim]).type(torch.uint8).detach().cpu()))
            cv2.imwrite(path, cv2.cvtColor(np.vstack(np.asarray(rows)), cv2.COLOR_BGR2RGB))


class MNIST_SVHN(BaseDataset):
    """Dataset class for the MNIST-SVHN bimodal dataset (can be also used for unimodal training)"""
    feature_dims = {"mnist": [28,28,1],
                    "svhn": [32,32,3]
                    }  # these feature_dims are also used by the encoder and decoder networks

    def __init__(self, pth, testpth, mod_type):
        super().__init__(pth, testpth, mod_type)
        self.mod_type = mod_type

    def _mod_specific_loaders(self):
        return {"mnist": self._process_mnist, "svhn": self._process_svhn}

    def _mod_specific_savers(self):
        return {"mnist": self._postprocess_mnist, "svhn": self._postprocess_svhn}

    def _postprocess_svhn(self, data):
        if isinstance(data, dict):
            data = data["data"]
        images = np.asarray(data.detach().cpu()).reshape(-1, *self.feature_dims["svhn"]) * 255
        images_res = []
        for i in images:
            images_res.append(cv2.resize(i, (28,28)))
        return np.asarray(images_res)

    def _postprocess_mnist(self, data):
        if isinstance(data, dict):
            data = data["data"]
        images = np.asarray(data.detach().cpu()).reshape(-1,*self.feature_dims["mnist"])*255
        images_3chan = cv2.merge((images, images, images)).squeeze(-2)
        return images_3chan

    def _process_mnist(self):
        return super(MNIST_SVHN, self)._preprocess_images([self.feature_dims["mnist"][i] for i in [2,0,1]])

    def _process_svhn(self):
        return super(MNIST_SVHN, self)._preprocess_images([self.feature_dims["svhn"][i] for i in [2,0,1]])

    def save_recons(self, data, recons, path, mod_names):
        output_processed = self._postprocess(recons)
        outs = add_recon_title(output_processed, "output\n{}".format(self.mod_type), (0, 170, 0))
        input_processed = []
        for key, d in data.items():
            output = self._mod_specific_savers()[mod_names[key]](d)
            images = add_recon_title(output, "input\n{}".format(mod_names[key]), (0, 0, 255))
            input_processed.append(np.vstack(images))
            input_processed.append(np.ones((np.vstack(images).shape[0], 2, 3))*125)
        inputs = np.hstack(input_processed).astype("uint8")
        final = np.hstack((inputs, np.vstack(outs).astype("uint8")))
        cv2.imwrite(path, cv2.cvtColor(final, cv2.COLOR_BGR2RGB))


class FASHION(BaseDataset):
    """Dataset class for the FashionMNIST dataset (for unimodal training)"""
    feature_dims = {"image": [28,28,1],
                    }  # these feature_dims are also used by the encoder and decoder networks

    def __init__(self, pth, testpth, mod_type):
        super().__init__(pth, testpth, mod_type)
        self.mod_type = mod_type
        self.labels_train = None

    def labels(self):
        return self.labels_train

    def get_data_raw(self):
        data = torchvision.datasets.FashionMNIST(root=self.path, train=True, download=True)
        self.labels_train = [int(x) for x in data.targets]
        return data.data.unsqueeze(-1)/255

    def _mod_specific_loaders(self):
        return {"image": self._process_fashionmnist}

    def _mod_specific_savers(self):
        return {"image": self._postprocess_fashionmnist}

    def _postprocess_fashionmnist(self, data):
        if isinstance(data, dict):
            data = data["data"]
        images = np.asarray(data.detach().cpu()).reshape(-1,*self.feature_dims["image"])*255
        images_3chan = cv2.merge((images, images, images)).squeeze(-2)
        return images_3chan

    def _process_fashionmnist(self):
        return super(FASHION, self)._preprocess_images([self.feature_dims["image"][i] for i in [2,0,1]])

    def save_recons(self, data, recons, path, mod_names):
        output_processed = self._postprocess(recons)
        outs = add_recon_title(output_processed, "output\n{}".format(self.mod_type), (0, 170, 0))
        input_processed = []
        for key, d in data.items():
            output = self._mod_specific_savers()[mod_names[key]](d)
            images = add_recon_title(output, "input\n{}".format(mod_names[key]), (0, 0, 255))
            input_processed.append(np.vstack(images))
            input_processed.append(np.ones((np.vstack(images).shape[0], 2, 3))*125)
        inputs = np.hstack(input_processed).astype("uint8")
        final = np.hstack((inputs, np.vstack(outs).astype("uint8")))
        cv2.imwrite(path, cv2.cvtColor(final, cv2.COLOR_BGR2RGB))

class EUROSAT(BaseDataset):
    """Dataset class for the EUROSAT dataset (for unimodal training)"""
    feature_dims = {"image": [64,64,3],
                    }  # these feature_dims are also used by the encoder and decoder networks

    def __init__(self, pth, testpth, mod_type):
        super().__init__(pth, testpth, mod_type)
        self.mod_type = mod_type
        self.labels_train = None

    def labels(self):
        return self.labels_train

    def get_data_raw(self):
        ssl._create_default_https_context = ssl._create_unverified_context
        data = torchvision.datasets.EuroSAT(root=self.path, download=True)
        paths = data.make_dataset(os.path.join(self.path, "eurosat/2750"), data.class_to_idx, extensions=[".jpg"])
        imgs, labels = [], []
        for p in paths:
            imgs.append(cv2.imread(p[0]))
            labels.append(p[1])
        self.labels_train = labels
        return np.asarray(imgs)

    def _mod_specific_loaders(self):
        return {"image": self._process_eurosat}

    def _mod_specific_savers(self):
        return {"image": self._postprocess_eurosat}

    def _postprocess_eurosat(self, data):
        if isinstance(data, dict):
            data = data["data"]
        return np.asarray(data.detach().cpu())*255

    def _process_eurosat(self):
        d = self.get_data_raw()
        return torch.tensor(d).reshape(-1, 3, 64, 64) / 255

    def save_recons(self, data, recons, path, mod_names):
        output_processed = self._postprocess_all2img(recons)
        outs = add_recon_title(output_processed, "output\n{}".format(self.mod_type), (0, 170, 0))
        input_processed = []
        for key, d in data.items():
            output = self._mod_specific_savers()[mod_names[key]](d)
            images = np.reshape(output,(-1,*self.feature_dims["image"]))
            images = add_recon_title(images, "input\n{}".format(mod_names[key]), (0, 0, 255))
            input_processed.append(np.vstack(images))
            input_processed.append(np.ones((np.vstack(images).shape[0], 2, 3))*125)
        inputs = np.hstack(input_processed).astype("uint8")
        final = np.hstack((inputs, np.vstack(outs).astype("uint8")))
        cv2.imwrite(path, final)

class SPRITES(BaseDataset):
    feature_dims = {"frames": [8,64,64,3],
                    "attributes": [4,6],
                    "actions": [9]
                    }  # these feature_dims are also used by the encoder and decoder networks

    def __init__(self, pth, testpth, mod_type):
        super().__init__(pth, testpth, mod_type)
        self.mod_type = mod_type
        self.text2img_size = (64, 145, 3)
        self.directions = ['front', 'left', 'right']
        self.actions = ['walk', 'spellcard', 'slash']
        self.label_map = ["walk front", "walk left", "walk right", "spellcard front", "spellcard left",
                          "spellcard right", "slash front", "slash left", "slash right"]
        self.attr_map = ["skin", "pants", "top", "hair"]
        self.att_names = [["pink", "yellow", "grey", "silver", "beige", "brown"], ["white", "gold", "red", "armor", "blue", "green"],
                          ["maroon", "blue", "white", "armor", "brown", "shirt"],["green", "blue", "yellow", "silver", "red", "purple"]]

    def labels(self):
        if self.current_path is None:
            return None
        actions = np.argmax(self.get_actions()[:, :9], axis=-1)
        labels = []
        for a in actions:
            labels.append(int(a))   #(self.label_map_int[int(a)])
        return labels

    def eval_statistics_fn(self):
        return sprites_eval

    def get_frames(self):
        X_train = []
        for act in range(len(self.actions)):
            for i in range(len(self.directions)):
                x = np.load(os.path.join(self.current_path, '{}_{}_frames_{}.npy'.format(self.actions[act], self.directions[i], self.current_datatype())))
                X_train.append(x)
        data = np.concatenate(X_train, axis=0)
        return torch.tensor(data)

    def get_attributes(self):
        self.categorical = True
        A_train = []
        for act in range(len(self.actions)):
            for i in range(len(self.directions)):
                a = np.load(os.path.join(self.current_path, '{}_{}_attributes_{}.npy'.format(self.actions[act], self.directions[i], self.current_datatype())))
                A_train.append(a[:, 0, :, :])
        data = np.concatenate(A_train, axis=0)
        return torch.tensor(data)

    def get_actions(self):
        self.categorical = True
        D_train = []
        for act in range(len(self.actions)):
            for i in range(len(self.directions)):
                a = np.load(os.path.join(self.current_path, '{}_{}_attributes_{}.npy'.format(self.actions[act], self.directions[i], self.current_datatype())))
                d = np.zeros([a.shape[0], 9])
                d[:, 3 * act + i] = 1
                D_train.append(d)
        data = np.concatenate(D_train, axis=0)
        return torch.tensor(data)

    def make_masks(self, shape):
        return torch.ones(shape).unsqueeze(-1)

    def _mod_specific_loaders(self):
        return {"frames": self.get_frames, "attributes": self.get_attributes, "actions": self.get_actions}

    def _mod_specific_savers(self):
        return {"frames": self._postprocess_frames, "attributes": self._postprocess_attributes,
                "actions": self._postprocess_actions}

    def _postprocess_frames(self, data):
        data = data["data"] if isinstance(data, dict) else data
        return np.asarray(data.detach().cpu().reshape(-1, *self.feature_dims["frames"])) * 255

    def _postprocess_actions(self, data):
        data = data["data"] if isinstance(data, dict) else data
        indices = np.argmax(data.detach().cpu(), axis=-1)
        return [self.label_map[int(i)] for i in indices]

    def _postprocess_attributes(self, data):
        data = data["data"] if isinstance(data, dict) else data
        indices = np.argmax(data.detach().cpu(), axis=-1)
        atts = []
        for i in indices:
            label = ""
            for att_i, a in enumerate(i):
                label += self.att_names[att_i][a] + " " + self.attr_map[att_i]
                label += " \n" if att_i in [0,1,3] else ", "
            atts.append(label)
        return atts

    def iter_over_inputs(self, outs, data, mod_names, f=0):
        input_processed = []
        for key, d in data.items():
            output = self._mod_specific_savers()[mod_names[key]](d)
            images = turn_text2image(output, img_size=self.text2img_size) if mod_names[key] in ["attributes", "actions"] \
                else output[:, f, :, :, :]
            images = add_recon_title(images, "input\n{}".format(mod_names[key]), (0, 0, 255))
            input_processed.append(np.vstack(images))
            input_processed.append(np.ones((np.vstack(images).shape[0], 2, 3)) * 145)
        inputs = np.hstack(input_processed).astype("uint8")
        return np.hstack((inputs, np.vstack(outs).astype("uint8")))

    def save_recons(self, data, recons, path, mod_names):
        output_processed = self._postprocess_all2img(recons)
        if self.mod_type != "frames" and [k for k, v in mod_names.items() if v == 'frames'][0] not in data.keys():
            outs = add_recon_title(output_processed, "output\n{}".format(self.mod_type), (0, 170, 0))
            final = self.iter_over_inputs(outs, data, mod_names)
            cv2.imwrite(path, cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
        else:
            timesteps = []
            for f in range(8):
                outs = add_recon_title(output_processed[:, f, :, :, :], "output\n{}".format(self.mod_type), (0, 170, 0))\
                if self.mod_type == "frames" else add_recon_title(output_processed, "output\n{}".format(self.mod_type), (0, 170, 0))
                final = self.iter_over_inputs(outs, data, mod_names, f)
                timesteps.append(final)
            imageio.mimsave(path.replace(".png", ".gif"), timesteps)

    def _postprocess_all2img(self, data):
        """
        Converts any kind of data to images to save traversal visualizations

        :param data: input data
        :type data: torch.tensor
        :return: processed images
        :rtype: torch.tensor
        """
        output_processed = self._postprocess(data)
        output_processed = turn_text2image(output_processed, img_size=self.text2img_size) \
            if self.mod_type in ["actions", "attributes"] else output_processed
        return output_processed

    def save_traversals(self, recons, path, num_dims):
        """
        Makes a grid of traversals and saves as animated gif image

        :param recons: data to save
        :type recons: torch.tensor
        :param path: path to save the traversal to
        :type path: str
        :param num_dims: number of latent dimensions
        :type num_dims: int
        """
        if self.mod_type != "frames":
            super().save_traversals(recons, path, num_dims)
        else:
            if len(recons.shape) < 4:
                grids = []
                output_processed = torch.tensor(self._postprocess_all2img(recons)).permute(1, 0, 4, 3, 2)
                for i in output_processed:
                    grids.append(
                        np.asarray(make_grid(i, padding=1, nrow=int(math.sqrt(len(recons)))).transpose(2, 0)).astype(
                            "uint8"))
                imageio.mimsave(path.replace(".png", ".gif"), grids)
            else:  # make traversal gifs
                output_processed = torch.stack([torch.tensor(self._postprocess_all2img(x)) for x in recons])
                output_processed = output_processed.reshape(num_dims, -1, *output_processed.shape[1:]).squeeze()
                rows = []
                for ind, dim in enumerate(output_processed):
                    rows.append(np.asarray(torch.hstack([x for x in dim]).type(torch.uint8).detach().cpu()))
                cv2.imwrite(path, cv2.cvtColor(np.vstack(np.asarray(rows)), cv2.COLOR_BGR2RGB))
