import torch
import numpy as np
import os
import logging
import argparse
from tensorboardX import SummaryWriter

from lib.model import minist_GAN as gan
from lib import data

import msvcrt
import time
import sys
import cv2

Z_SPACE_DIM = 10
SAVE_DIR = "./saves"


def input_with_timeout(prompt, timeout, timer=time.monotonic):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    endtime = timer() + timeout
    result = []
    while timer() < endtime:
        if msvcrt.kbhit():
            # XXX can it block on multibyte characters?
            result.append(msvcrt.getwche())
            if result[-1] == '\n':  # XXX check what Windows returns here
                return ''.join(result[:-1])
        time.sleep(0.04)  # just to yield to other processes/threads
    print("")
    return ""


def split(images, labels):
    data = {}
    for img, l in zip(images, labels):
        if l not in data.keys():
            data[l] = []
        data[l].append(img)
    return data


def generate_z(label, num, dim=Z_SPACE_DIM):
    noise = torch.rand(num)
    z = torch.rand(num, dim)
    z[:, label] = noise
    return z


def show(generator, device):

    for _ in range(100):
        z = torch.rand(2, Z_SPACE_DIM).to(device=device)
        sample_gz = generator(z).cpu().detach().numpy()[0]
        sample_gz = sample_gz.reshape([28, 28, 1])
        sample_gz = sample_gz*255
        sample_gz = sample_gz.astype(np.uint8)
        sample_gz = cv2.resize(sample_gz, (256, 256))
        cv2.imshow("image", sample_gz)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()


def adjust_learning_rate(optimizer, lr=1e-3):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(args):
    os.makedirs(SAVE_DIR, exist_ok=True)
    x, y = data.load_mnist(args.data)
    test_x, test_y = data.load_mnist(args.data, "t10k")
    features_len = x.shape[-1]
    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    generator = gan.Generator(in_features=Z_SPACE_DIM,
                              out_features=features_len).to(device)

    discriminator = gan.Discriminator(
        in_features=features_len, out_features=1).to(device)

    if args.load:
        model_dict = torch.load(args.load)
        generator.load_state_dict(model_dict["generator"])
        discriminator.load_state_dict(model_dict["discriminator"])

    if args.show:
        show(generator, device)
        return

    iter_num = 100000
    k_steps = 1
    m = 1000
    start = 0
    name = "train"
    log = logging.getLogger(name)
    writer = SummaryWriter(comment="-"+name)

    x = torch.tensor(x, dtype=torch.float32).to(device)
    x = (x)/255
    test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
    test_x = (test_x)/255

    num = x.shape[0]
    m = min(num, m)
    test_num = test_x.shape[0]

    lr = 1e-2
    discriminator_optimizer = torch.optim.SGD(
        params=discriminator.parameters(), lr=lr)
    generator_optimizer = torch.optim.SGD(
        params=generator.parameters(), lr=lr)
    
    while True:
        for i in range(start, iter_num+start):
            discriminator_loss = []
            for k in range(k_steps):
                idxes = torch.randint(0, num, [m]).to(device=device)
                sample_x = x.index_select(0, idxes)
                z = torch.rand(m, Z_SPACE_DIM).to(device=device)
                sample_gz = generator(z).detach()

                Dx = discriminator(sample_x)
                Dgz = discriminator(sample_gz)

                Dx = torch.clamp(Dx, min=1e-5, max=0.99999)
                Dgz = torch.clamp(Dgz, min=1e-5, max=0.99999)

                loss = torch.log(Dx) + torch.log(1-Dgz)

                loss = - torch.mean(torch.sum(loss, -1))
                discriminator_optimizer.zero_grad()
                loss.backward()
                discriminator_loss.append(loss.item())
                discriminator_optimizer.step()

            writer.add_scalar("discriminator_loss", np.mean(
                discriminator_loss), global_step=i)

            z = torch.rand(m, Z_SPACE_DIM).to(device=device)
            sample_gz = generator(z)
            Dgz = discriminator(sample_gz)
            Dgz = torch.clamp(Dgz, min=1e-5, max=0.99999)
            loss = torch.log(Dgz)
            loss = -torch.mean(torch.sum(loss, -1))
            generator_optimizer.zero_grad()
            loss.backward()
            generator_optimizer.step()

            writer.add_scalar("generator_loss", loss.item(), global_step=i)

            if i % 1000 == 0:
                z = torch.rand(test_num, Z_SPACE_DIM).to(device=device)
                sample_gz = generator(z)
                Dx = discriminator(test_x)
                Dgz = discriminator(sample_gz)
                TP = float(torch.sum(Dx > 0.5))
                TN = float(torch.sum(Dgz < 0.5))
                accuracy = (TP+TN)/(2*test_num)
                recall = TP/test_num
                specificity = TN/test_num
                writer.add_scalar("test_accuracy", accuracy, global_step=i)
                writer.add_scalar("test_recall", recall, global_step=i)
                writer.add_scalar("test_specificity",
                                  specificity, global_step=i)
                log.info("Epoch: %d, test_accuracy: %f, test_recall: %f, test_specificy: %f" % (
                    i, accuracy, recall, specificity))
                torch.save({"generator": generator.state_dict(),
                            "discriminator": discriminator.state_dict()},
                           os.path.join(SAVE_DIR, "Epoch_%d_accuracy_%.3f_recall_%.3f_specificy_%.3f" % (i, accuracy, recall, specificity)))
                
                adjust_learning_rate(generator_optimizer, lr)
                adjust_learning_rate(discriminator_optimizer, lr)
                lr = max(lr*0.9999, 1e-3)
                writer.flush()

        still_train = input_with_timeout("continue?[yes]: ", 10)
        start = i
        if still_train != "" and still_train == "no":
            break


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s",
                        level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, help="data_path")
    parser.add_argument("--cuda", action="store_true",
                        default=False, help="using cuda")
    parser.add_argument("-l", "--load", default=False,
                        required=False, help="model path")
    parser.add_argument("-s", "--show", default=False, action="store_true",
                        required=False, help="show generated images")
    args = parser.parse_args()
    main(args)
