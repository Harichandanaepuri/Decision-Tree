if __name__ == '__main__':
    with open("tic-tac-toe.data", "r") as filestream:
        with open("dataset_tic_tac_toe.data","w") as f2:
            for currentline in filestream:
                #print(currentline)
                temp =""
                tokens=currentline.split(",")
                print(tokens)
                for i in tokens:
                    if (i=='x'):
                        temp=temp+'0'+','
                    if(i=='b'):
                        temp = temp + '1'+','
                    if(i=='o'):
                        temp = temp + '2'+','
                    if(i=="negative\n"):
                        temp = temp + '3'
                    if(i=="positive\n"):
                        #temp.append((1))
                        temp = temp + '4'
                print(temp)
                f2.write(temp)
                f2.write('\n')






