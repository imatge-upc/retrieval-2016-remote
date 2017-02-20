
   for i = 1:420 %420
       
     %c = flipud((sort(r(i,:))))
     [val idx] = sort(r(i,:),'ascend')
     %result = c(401:420); % top 20  %2081:2100  
     % index of 20 most similar images
     %ind(i,:) = (find(r(i,:)>= c(401),20) )
     ind(i,:) =idx(1:20) % (1:10)
     %ind = fliplr(ind)
   
   end
   %%
    excel = xlsread('Multi_labels.xlsx','A2:R421') 
   % create prediction matrix
   for j = 1:420 %420
      rec(j) = 0
      pre(j) = 0
      acc(j) = 0
      hl(j)  = 0
      for i = 1:20 % 1:20
         
       n(i) = ind(j,i)    
       prediction(i,:) = excel(n(i),:)
       % compute matrics based on multi-labels information
       rec(j)= rec(j) + fHSI_CBIR_recall(excel(j,:),prediction(i,:))
       pre(j)= pre(j) + fHSI_CBIR_precision(excel(j,:),prediction(i,:))
       acc(j)= acc(j) + fHSI_CBIR_accuracy(excel(j,:),prediction(i,:))
       hl(j) = hl(j) + fHSI_CBIR_hamming_loss(excel(j,:),prediction(i,:))
       if i == 20
       rec_media_new(j) = rec(j)/20
       pre_media_new(j) = pre(j)/20
       acc_media_new(j) = acc(j)/20
       hl_media_new(j) = hl(j)/20
       end
      end
   end
   %%
    excel = xlsread('Classes_labels.xlsx','B2:B421') 
   % create prediction matrix
   for j = 1:420
      rec(j) = 0
      pre(j) = 0
      acc(j) = 0
      hl(j)  = 0
      for i = 1:20
         
       n(i) = ind(j,i)    
       prediction(i) = excel(n(i))
       % compute matrics based on multi-labels information
       %rec(j)= rec(j) + fHSI_CBIR_recall(excel(j),prediction(i))
       %pre(j)= pre(j) + fHSI_CBIR_precision(excel(j),prediction(i))
       acc(j)= acc(j) + (excel(j) == prediction(i))
       %hl(j) = hl(j) + fHSI_CBIR_hamming_loss(excel(j),prediction(i))
       if i ==20
       %rec_media_new(j) = rec(j)/20
       %pre_media_new(j) = pre(j)/20
       acc_media_new(j) = acc(j)/20
       %hl_media_new(j) = hl(j)/20
       end
      end
   end
     %% %
  [numData textData rawData] = xlsread('LandUse_Multilabeled_update_1680.xlsx','A2:R1681')
  save('pqfile.txt', 'numData','-ASCII');