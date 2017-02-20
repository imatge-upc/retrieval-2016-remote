while (~feof(fileID))                               % For each block:                         
   Block=1
   fprintf('Block: %s\n', num2str(Block))           % Print block number to the screen
   InputText = textscan(fileID,'%s',2,'delimiter','\n');  % Read 2 header lines
   HeaderLines{Block,1} = InputText{1};
   disp(HeaderLines{Block});                        % Display header lines
   
   InputText = textscan(fileID,'Num SNR = %f');     % Read the numeric value 
                                                    % following the text, Num SNR =
   NumCols = InputText{1};                          % Specify that this is the 
                                                    % number of data columns
   
   FormatString = repmat('%f',1,NumCols);           % Create format string
                                                    % based on the number
                                                    % of columns
   InputText = textscan(fileID,FormatString, ...    % Read data block
      'delimiter',',');
   
   Data{Block,1} = cell2mat(InputText);              
   [NumRows,NumCols] = size(Data{Block});           % Determine size of table
   disp(cellstr(['Table data size: ' ...
      num2str(NumRows) ' x ' num2str(NumCols)]));
   disp(' ');                                       % New line
   
   eob = textscan(fileID,'%s',1,'delimiter','\n');  % Read and discard end-of-block marker 
   Block = Block+1;                                 % Increment block index
end