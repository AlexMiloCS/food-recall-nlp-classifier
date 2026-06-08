import time
import copy
import torch
import numpy as np
from sklearn.metrics import f1_score

def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, criterion_hazard, criterion_product, device, epochs):
    best_st1_score = 0.0
    best_model_weights = None

    for epoch in range(epochs):
        print(f"\n{'='*30}\nEpoch {epoch+1}/{epochs}\n{'='*30}")

        # Training Phase
        model.train()
        total_train_loss = 0
        start_time = time.time()

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            hazard_labels = batch['hazard_label'].to(device)
            product_labels = batch['product_label'].to(device)

            hazard_logits, product_logits = model(input_ids, attention_mask)

            loss_hazard = criterion_hazard(hazard_logits, hazard_labels)
            loss_product = criterion_product(product_logits, product_labels)

            loss = (0.7 * loss_hazard) + (0.3 * loss_product)

            loss.backward()
            
            # Gradient clipping is applied here safely
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Step the scheduler only if it exists (useful for switching between BERT and RoBERTa)
            if scheduler is not None:
                scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Εκπαιδευση: {(time.time() - start_time)/60:.2f} λεπτα | Avg Loss: {avg_train_loss:.4f}")

        # Validation Phase
        model.eval()
        all_hazard_preds, all_product_preds, all_hazard_true, all_product_true = [], [], [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                hazard_logits, product_logits = model(input_ids, attention_mask)

                all_hazard_preds.extend(torch.argmax(hazard_logits, dim=1).cpu().numpy())
                all_product_preds.extend(torch.argmax(product_logits, dim=1).cpu().numpy())
                all_hazard_true.extend(batch['hazard_label'].cpu().numpy())
                all_product_true.extend(batch['product_label'].cpu().numpy())

        f1_hazard = f1_score(all_hazard_true, all_hazard_preds, average='macro')

        all_hazard_preds = np.array(all_hazard_preds)
        all_hazard_true = np.array(all_hazard_true)
        all_product_preds = np.array(all_product_preds)
        all_product_true = np.array(all_product_true)

        correct_mask = (all_hazard_preds == all_hazard_true)
        if correct_mask.sum() == 0:
            f1_product_cond = 0.0
        else:
            f1_product_cond = f1_score(all_product_true[correct_mask], all_product_preds[correct_mask], average='macro')

        st1_score = (f1_hazard + f1_product_cond) / 2

        print(f"Validation -> Hazard F1: {f1_hazard:.4f} | Product F1: {f1_product_cond:.4f} | ST1 Score: {st1_score:.4f}")

        # Checkpointing
        if st1_score > best_st1_score:
            best_st1_score = st1_score
            best_model_weights = copy.deepcopy(model.state_dict())
            print("Νεο ρεκορ. Τα βαρη αποθηκευτηκαν.")

    # Load the best weights back into the model before returning
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    print(f"\nΗ εκπαιδευση ολοκληρωθηκε. Κορυφαιο ST1: {best_st1_score:.4f}")
    
    return model