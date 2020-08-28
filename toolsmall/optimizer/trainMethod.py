import time
import torch
import logging
import numpy as np
import os

def forwardV1(network,train,lr_scheduler,epochs,save_model):
    for epoch in range(epochs):
        learning_rate = lr_scheduler.get_last_lr()[0]
        start_time = time.time()
        mean_loss = train(epoch)
        end_time = time.time()
        # update the learning rate
        lr_scheduler.step()
        # save model
        # torch.save(network.state_dict(), save_model)
        torch.save({'model': network.state_dict(), "epoch": epoch, 'lr': learning_rate}, save_model)


        logging.info("\nepoch:%d learning_rate:%f mean_loss:%f cost_time:%f\n" %
              (epoch, learning_rate, mean_loss, end_time - start_time))

def forwardV2(network,train,optimizer,epochs,save_model,lr):
    init_mean_loss = np.inf
    best_save_model = None
    best_epoch = 0
    gamma = 0.8
    last_lr = lr / 50
    for epoch in range(epochs):
        if epoch - best_epoch > 5:
            init_mean_loss = np.inf
            for param_group in optimizer.param_groups:
                learning_rate = param_group['lr']
                param_group['lr'] = max(learning_rate * gamma, last_lr)
        else:
            for param_group in optimizer.param_groups:
                learning_rate = param_group['lr']
                param_group['lr'] = max(learning_rate, last_lr)

        start_time = time.time()
        mean_loss = train(epoch)
        end_time = time.time()

        logging.info("\nepoch:%d learning_rate:%f mean_loss:%f cost_time:%f\n" %
                     (epoch, learning_rate, mean_loss, end_time - start_time))

        torch.save({'model': network.state_dict(), "epoch": epoch, 'lr': learning_rate}, save_model)

        # 保留效果最好的参数
        if mean_loss < init_mean_loss:
            init_mean_loss = mean_loss
            if best_save_model is not None:
                # 删除上一次保持的
                os.remove(best_save_model)
            best_save_model = save_model.replace(".pth", "_%s.pth" % (str(epoch)))
            # torch.save(self.network.state_dict(), best_save_model)
            torch.save({'model': network.state_dict(), "epoch": epoch, 'lr': learning_rate}, best_save_model)
            best_epoch = epoch

def forwardV3(network, train, lr_scheduler, epochs, save_model):
    init_mean_loss = np.inf
    best_save_model = None
    best_epoch = 0
    for epoch in range(epochs):
        if epoch - best_epoch > 5:
            init_mean_loss = np.inf
            lr_scheduler.step()
        learning_rate = lr_scheduler.get_last_lr()[0]

        start_time = time.time()
        mean_loss = train(epoch)
        end_time = time.time()

        logging.info("\nepoch:%d learning_rate:%f mean_loss:%f cost_time:%f\n" %
                     (epoch, learning_rate, mean_loss, end_time - start_time))

        torch.save({'model': network.state_dict(), "epoch": epoch, 'lr': learning_rate}, save_model)

        # 保留效果最好的参数
        if mean_loss < init_mean_loss:
            init_mean_loss = mean_loss
            if best_save_model is not None:
                # 删除上一次保持的
                os.remove(best_save_model)
            best_save_model = save_model.replace(".pth", "_%s.pth" % (str(epoch)))
            # torch.save(self.network.state_dict(), best_save_model)
            torch.save({'model': network.state_dict(), "epoch": epoch, 'lr': learning_rate}, best_save_model)
            best_epoch = epoch
