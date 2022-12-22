from utils import adjust_learning_rate, AverageMeter,update_or_add, extract_query_or_gall_feat
from torch.autograd import Variable
import time
from evaluation import evaluation, eval_sysu
import numpy as np

def train(epoch, lr, net, reid, trainloader, device, optimizer, criterion_id, criterion_tri,
          log_file_name, Curves, scheduler, model_name, save_files, Augmix, XPATCH, XREA, ML_MDA):

    if model_name == "transreid" :
        scheduler.step(epoch)
        print(f"Current lr : {scheduler._get_lr(epoch)[0]}")
    elif model_name == "LMBN":
        epoch = scheduler.last_epoch
        print(f"Current lr : {scheduler.get_last_lr()[0]}")
        scheduler.step()
    else:
        current_lr = adjust_learning_rate(optimizer, epoch, lr=lr)


    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()

    data_time = AverageMeter()
    batch_time = AverageMeter()

    correct = 0
    total = 0



    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

        # Labels 1 and 2 are the same because the two inputs correspond to the same identity
        labels = label1

        input1 = Variable(input1.to(device))
        input2 = Variable(input2.to(device))
        labels = Variable(labels.to(device))

        data_time.update(time.time() - end)

        if model_name == "transreid":
            out0, feat = net(input1)
        elif model_name == "LMBN":
            out0 = net(input1) # Out regroup is : (list of out0, list of feat)
        else:
            feat, out0 = net([input1, input2], model_name, reid)

        #### LOSS CALCULATION ####
        if (Augmix or XPATCH!="False" or XREA!="False" or ML_MDA):
            loss_ce, loss_tri = criterion_id(out0, feat, labels)
        elif model_name == "transreid":
            loss_ce, loss_tri = [], []
            # Calcul losses for each global or local features (5 feat)
            for featL, out0L in zip(feat, out0):
                temp_loss_ce, temp_loss_tri = criterion_id(out0L, featL, labels)
                loss_ce.append(temp_loss_ce)
                loss_tri.append(temp_loss_tri)
            # Global feat count for 0.5 and locals for 05
            loss_ce = 0.5 * sum(loss_ce[1:]) / (len(loss_ce) - 1) + 0.5 * loss_ce[0]
            loss_tri = 0.5 * sum(loss_tri[1:]) / (len(loss_tri) - 1) + 0.5 * loss_tri[0]
        elif model_name == "LMBN":
            loss_ce = criterion_id.compute(out0, labels)
        else:
            loss_ce = criterion_id(out0, labels)
            loss_tri, batch_acc = criterion_tri(feat, labels)

        try:
            _, predicted = out0.max(1)
        except:
            try:
                _, predicted = out0[0].max(1)
            except:
                try:
                    _, predicted = out0[0][0].max(1)
                except:
                    pass

        if (Augmix or XPATCH!="False" or XREA!="False" or ML_MDA) \
                or model_name in ["transreid", "LMBN"] :
            correct += (predicted.eq(labels).sum().item())
        else :
            correct += (batch_acc / 2)
            correct += (predicted.eq(labels).sum().item() / 2)

        loss = loss_ce
        if model_name != "LMBN":
            loss += loss_tri

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))

        id_loss.update(loss_ce.item(), 2 * input1.size(0))
        if model_name != "LMBN":
            tri_loss.update(loss_tri.item(), 2 * input1.size(0))

        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        values_to_display = []
        if batch_idx % 50 == 0:
            values_to_display.append(f'Epoch: [{epoch}][{batch_idx}/{len(trainloader)}] ')
            values_to_display.append(f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) ')
            values_to_display.append(f'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f}) ')
            values_to_display.append(f'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) ')
            values_to_display.append(f'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) ')
            values_to_display.append(f'Accu: {100. * correct / total:.2f} ')
            to_disp = ""
            for value in values_to_display:
                to_disp += value
            print(to_disp)
            if save_files:
                with open(log_file_name, 'a') as log_file:
                    log_file.write(to_disp + "\n")

    update_or_add(dico=Curves, key="Overall_loss", value=train_loss.avg)
    update_or_add(dico=Curves, key="Cross entropy", value=id_loss.avg)
    update_or_add(dico=Curves, key="Triplet losses", value=tri_loss.avg)
    update_or_add(dico=Curves, key="Accuracy", value=100. * correct / total)

def valid(dataset, query_cam, gall_cam, query_label, gall_label, gall_loader, query_loader, net, model_name, reid,
      n_query, n_gall, Curves={}):
    # Get the timer
    end = time.time()

    ### Get all distances
    gall_feat_pool, gall_feat_fc, _ = extract_query_or_gall_feat(gall_loader, n_gall, net, modality=reid, QorG="Gallery", model=model_name)
    query_feat_pool, query_feat_fc, _ = extract_query_or_gall_feat(query_loader, n_query, net, modality=reid, QorG="Query", model=model_name)

    print(f"Feature extraction time : {time.time() - end}")
    start = time.time()

    ### Similarity (cosine)
    distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
    distmat_fc = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

    ### Evaluation
    if dataset == 'RegDB' or dataset == 'TWorld':
        cmc, mAP, mINP = evaluation(-distmat_fc, query_label, gall_label)
        cmc_att, mAP_att, mINP_att = evaluation(-distmat_pool, query_label, gall_label)

    elif dataset == 'SYSU':
        cmc, mAP, mINP = eval_sysu(-distmat_fc, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)

    ### CURVES MANAGEMENT
    keys = ["Validation_mAP_BN", "Validation_mINP_BN", "Validation_mAP", "Validation_mINP"]
    values = [mAP, mINP, mAP_att, mINP_att]

    for key, value in zip(keys, values):
        update_or_add(dico=Curves, key=key, value=value)

    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att