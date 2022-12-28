import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import cv2
import matplotlib as mpl


def calc_cossim(features1, features2):
    assert features1.ndim == 2
    assert features2.ndim == 2
    f1 = features1 / np.linalg.norm(features1, 2, -1, keepdims=True)
    f2 = features2 / np.linalg.norm(features2, 2, -1, keepdims=True)
    cossims = (f1 * f2).sum(-1)
    return cossims


def make_similarity_plot(save_path,
                         probe_features, probe_weights, probe_fused_feature, probe_image_list,
                         gallery_feature, gallery_image):
    indv_similarities = calc_cossim(probe_features, gallery_feature)
    fused_similarity = calc_cossim(np.expand_dims(probe_fused_feature, 0), gallery_feature)
    draw_similarity_plot(save_path, indv_similarities, fused_similarity,
                         transparency=probe_weights,
                         images=probe_image_list, ref_image=np.array(gallery_image))


def draw_similarity_plot(save_path, similarities, special_values, transparency=[], images=[], ref_image=None):
    np.random.seed(2)

    if ref_image is not None:
        x = [['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B']]
    else:
        x = [['A']]
    fig, axes = plt.subplot_mosaic(mosaic=x, figsize=(30, 5))
    ax = axes['A']

    marker = 'o'
    size = 20
    plt.rcParams.update({'font.size': 20})

    if len(transparency) > 0:
        assert len(transparency) == len(similarities)
        a_max = 1.0
        a_min = 0.0
        alphas = transparency - transparency.min()
        alphas = alphas / alphas.max()
        alphas = alphas * (a_max - a_min)
        alphas = alphas + a_min
    else:
        alphas = np.ones_like(similarities)

    # plot individual points
    y_coords = []
    for idx, value in enumerate(similarities):
        # make color with transparency
        color = plt.cm.jet(alphas[idx])
        y_coord = np.random.uniform(-1, 1)
        ax.scatter(value, y_coord, s=size * 5, marker=marker, color=color, alpha=0.8)
        y_coords.append(y_coord)

    ax.scatter(special_values, 0, s=size * 50, marker="*", color='black', alpha=0.8)
    if 'average' in save_path:
        tag = 'AVG'
    elif 'cluster_and_aggregate' in save_path:
        tag = "CAFace"
    else:
        raise ValueError('not implemented yet')
    ax.annotate(tag, (special_values[0], 0), fontsize=15, va='center', xytext=(18, 0), textcoords="offset points")

    # set axis label
    ax.set_xlabel('Cosine Similarity between Gallery and Individual Probe', fontdict={'size': 20})
    ax.set_yticks([])
    # ax.set_xticks(ax.get_xticks(), fontsize=100)
    ax.xaxis.set_tick_params(labelsize=20)

    # set color bar
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=transparency.min(), vmax=transparency.max())
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 orientation='vertical', label='Sample Weights', location='left', pad=0.01)

    if len(images) > 0:
        assert len(images) == len(similarities)
        shown_images = np.array([[1., 1.]])  # just something big
        for x_value, y_coord, img_path, alpha in zip(similarities, y_coords, images, alphas):
            image_y = 2.0
            img_coord = np.array([x_value, image_y])
            dist = np.sum((img_coord - shown_images) ** 2, 1)
            if np.min(dist) < 5e-4:
                ## don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [img_coord]]

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tiny = cv2.resize(img, (64, 64))
            color = plt.cm.jet(alpha)
            # add line
            ax.plot([x_value, x_value], [image_y, y_coord], c=color, alpha=0.8)
            # add image
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(img_tiny, cmap=plt.cm.gray_r),
                img_coord,
                bboxprops=dict(edgecolor=color))
            ax.add_artist(imagebox)
    plt.title(' ', fontdict={'size': 100})

    if ref_image is not None:
        axes['B'].imshow(ref_image)
        axes['B'].set_yticks([])
        axes['B'].set_xticks([])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    plt.cla()

