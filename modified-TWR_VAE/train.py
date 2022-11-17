import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



def get_sentence(corpus, batch):
    sens = []
    for b in range(batch.shape[1]):
        sen = [corpus.id_word[batch[i, b].item()]
               for i in range(batch.shape[0])]
        sens.append(" ".join(sen))
    return sens


def sentence_acc(prod, target):
    target = target[1:]
    mask = target == 0
    prod = prod.argmax(dim=2)
    prod[mask] = -1
    correct = torch.eq(prod, target).to(dtype=torch.float).sum()
    return correct.item()

def train(encoder, decoder, opt, device, corpus, ep, prior_mu, prior_logvar, alpha, beta, gamma=0):
    
    encoder.train()
    decoder.train()
    total = 0
    recon_loss_total = 0
    kl_loss_total = 0
    correct_total = 0
    words_total = 0
    batch_total = 0
    for i, sen in enumerate(corpus):
        # sen: [len_sen, batch]
        batch_size = sen.shape[1]
        opt.zero_grad()
        total += sen.shape[1]
        sen = sen.to(device)
        z, mu, logvar, sen_len = encoder(sen)

        prod = decoder(z, sen)
        kl_loss = encoder.loss(mu, logvar, prior_mu, prior_logvar, alpha, beta, gamma)
        recon_loss = decoder.loss(prod, sen, sen_len)
        
        ((kl_loss+recon_loss)*1).backward()
        opt.step()
        
        recon_loss_total = recon_loss_total + recon_loss.item()
        kl_loss_total = kl_loss_total + kl_loss.item()
        correct = sentence_acc(prod, sen)
        words = sen_len.sum().item()
        correct_total = correct_total + correct
        words_total = words_total + words
        batch_total += batch_size
        
    print(
        f"\nTrain {ep}: recon_loss={(recon_loss_total/(batch_total)):.04f}, div_loss={(kl_loss_total/(batch_total)):.04f}, nll_loss={((recon_loss_total+kl_loss_total)/(batch_total)):.04f}, nll_loss_perword={((recon_loss_total+kl_loss_total)/words_total):.04f}, ppl={(np.exp((recon_loss_total+kl_loss_total)/words_total)):.04f}, acc={(correct_total/words_total):.04f}")
    return recon_loss_total/(batch_total), kl_loss_total/(batch_total),  correct_total/words_total, (recon_loss_total+kl_loss_total)/(batch_total), np.exp((recon_loss_total+kl_loss_total)/words_total)

# =================================




def reconstruction(encoder, decoder, opt, device, corpus, raw_corpus, recon_dir, ep, prior_mu, prior_logvar, alpha, beta, gamma=0):
    encoder.eval()
    decoder.eval()
    out_org = []
    out_recon_mu = []
    recon_loss_total = 0
    kl_loss_total = 0
    words_total = 0
    batch_total = 0
    
    for i, sen in enumerate(corpus):
        b_size = sen.shape[1]
        out_org += get_sentence(raw_corpus,sen[1:])
        sen = sen.to(device)
        
        with torch.no_grad():
            z, mu, logvar, sen_len = encoder(sen)
        
            recon_mu = decoder(z, sen)
            kl_loss = encoder.loss(mu, logvar, prior_mu, prior_logvar, alpha, beta, gamma)
            recon_loss = decoder.loss(recon_mu, sen, sen_len)
            
            sens_mu = recon_mu.argmax(dim=2)
            out_recon_mu += get_sentence(raw_corpus, sens_mu.to("cpu"))

            recon_loss_total = recon_loss_total + recon_loss.item()
            kl_loss_total = kl_loss_total + kl_loss.item()
            
            words = sen_len.sum().item()
            
            words_total = words_total + words
            batch_total += b_size
    
    recon_loss = recon_loss_total/batch_total
    kl_loss = kl_loss_total/batch_total
    nll_loss = (recon_loss_total+kl_loss_total)/batch_total
    nll_loss_perword = (recon_loss_total+kl_loss_total)/words_total
    ppl = np.exp((recon_loss_total+kl_loss_total)/words_total)
    
    
    print(f"Test: recon_loss:{recon_loss:.04f}, kl_loss:{kl_loss:.04f}, nll_loss:{nll_loss:.04f}, nll_loss_perword:{nll_loss_perword:.04f}, ppl:{ppl:.04f}")

    
    text = []
    if ep % 10 == 0:
        for i in range(len(out_recon_mu)):
            text.append("origin: " + out_org[i])
            text.append("reco_mu: " + out_recon_mu[i])
            text.append("\n")
        with open(recon_dir+f"TWRvae_outcome_{ep}.txt", "w") as f:
            f.write("\n".join(text))
            
    with open(recon_dir+f"test_TWRvae_loss.txt", "a") as f:
        f.write(f"{ep}\t{recon_loss}\t{kl_loss}\t{nll_loss}\t{ppl}\n")
    
    return ppl
