from config import *
from sklearn.linear_model import LogisticRegression
from wass_loss import *
from wasserstein import *


def fit_log_reg(X, y, random_state=0):
    clf = LogisticRegression(random_state=random_state).fit(X, y)
    ic(clf.score(X, y))
    return clf


def dset_to_x_y(dset, n):
    return torch.stack([x[0] for x in dset])[0:n], torch.tensor([x[1] for x in dset])[0:n]


def plot_lr_result(x, y_m, y_fm):
    plt.plot(x, y_m, marker='o', label="MNIST")
    plt.plot(x, y_fm, marker='s', label="FashionMNIST")
    plt.plot(x, (y_m + y_fm) / 2, marker='x', label="Total")
    plt.legend()
    # plt.xlim(xmin=10)
    # plt.ylim(ymin=0.1, ymax=0.7)
    plt.xlabel("Number of OoD Samples Seen")
    plt.xticks(x, x)
    plt.title("Detection Accuracy vs. Number of OoD Samples")
    plt.savefig("checkpoint/MNIST-FashionMNIST/lr.png")
    plt.close()


# TODO: refactor this
# def train_test(num_exposed, n, n_test):
#     # num_exposed = ['mnist128']
#     # n = 1000
#     # n_test = 2000
#     for stamp in num_exposed:
#         print(f"Number of OoD Samples Seen: " + stamp)
#         print(f"Number of Generated & InD Training Samples: {n}:{n}")
#         print(f"Number of OoD & InD Unseen Testing Samples: {n_test}:{n_test}")
#         # Create D and G
#         D = dc_discriminator(img_info, GAN_TYPE.OOD).to(DEVICE)
#         G = dc_generator().to(DEVICE)
#         # Load checkpoints
#         gan_checkpoint = GOOGLE_DRIVE_PATH + \
#             '/checkpoint/MNIST-FashionMNIST/' + stamp + '.pt'
#         checkpoint_dict = {'addr': gan_checkpoint, 'id': stamp}
#         chpt = torch.load(checkpoint_dict['addr'])
#         D.load_state_dict(chpt['D-state'])
#         G.load_state_dict(chpt['G-state'])
#         logger = chpt['logger']
#         print(f"Checkpoint [{checkpoint_dict['id']}] loaded.")
#         # Generating data points
#         g_seed = sample_noise(n, 96)
#         x_oodg = G(g_seed).cpu()
#         m, my = dset_to_x_y(mnist_tri_set, n)
#         # Test data points
#         fm_val, fmy_val = dset_to_x_y(fm_val_set, n_test)
#         m_val, my_val = dset_to_x_y(mnist_val_set, n_test)

#         # Convert to Wass Distance
#         w_m = ood_wass_loss(torch.softmax(D(m.to(DEVICE)), dim=-1))
#         w_oodg = ood_wass_loss(torch.softmax(D(x_oodg.to(DEVICE)), dim=-1))
#         print(f"Mean w_m {torch.mean(w_m)} ; Mean w_oodg {torch.mean(w_oodg)}")
#         x = torch.cat([w_m, w_oodg])
#         y = torch.ones(2*n)
#         y[0:n] = 0
#         # Train Logistic Regression Model
#         X = x.unsqueeze(-1).data.cpu()
#         y = y.data.cpu()
#         clf = LogisticRegression(random_state=0).fit(X, y)
#         print(f"Training accuracy: {clf.score(X,y)}")
#         # Evaluation
#         w_fm_val = ood_wass_loss(torch.softmax(D(fm_val.to(DEVICE)), dim=-1))
#         w_m_val = ood_wass_loss(torch.softmax(D(m_val.to(DEVICE)), dim=-1))
#         print(
#             f"Testing accuracy on FashionMNIST: {clf.score(w_fm_val.unsqueeze(-1).data.cpu(), np.ones(n_test))}")
#         print(
#             f"Testing accuracy on MNIST: {clf.score(w_m_val.unsqueeze(-1).data.cpu(), np.zeros(n_test))}")


if __name__ == '__main__':
    ic("Logistic Regression")
    n = np.array([8, 16, 32, 64, 128, 256])
    y_m = np.array([0.96733, 0.9743, 0.9766, 0.967, 0.947, 0.9306])
    y_fm = np.array([0.93556, 0.9864, 0.9826, 0.9926, 0.9953, 0.9999])
    plot_lr_result(n, y_m, y_fm)
