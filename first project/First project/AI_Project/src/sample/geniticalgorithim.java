package sample;

import java.util.*;

public class geniticalgorithim {
    ArrayList<info> arr;
    private ArrayList<int[]> arr1;

    public geniticalgorithim(ArrayList<int[]> arr1, ArrayList<info> arr) {
        this.arr1 = arr1;
        this.arr = arr;
    }

    //fitness function
    public int fitness_fun(int[] arr1) {
        int sum = 0;
        for (int j = 0; j < arr.size(); j++) {
            if (arr1[j] == arr.get(j).getF())
                sum += 111;
            else if (arr1[j] == arr.get(j).getS())
                sum += 74;
            else if (arr1[j] == arr.get(j).getT())
                sum += 37;
        }
        return sum;
    }


    //for sorting the list of population based on fitness function
    public ArrayList<int[]> bubblesort() {
        int n = arr1.size();
        for (int i = 0; i < n - 1; i++)
            for (int j = 0; j < n - i - 1; j++)
                if (fitness_fun(arr1.get(j)) < fitness_fun(arr1.get(j + 1))) {
                    // swap arr[j+1] and arr[j]
                    Collections.swap(arr1, j, j + 1);

                }
        return arr1;
    }

    //cross over function
    public ArrayList<int[]> cross_over() {
        //creat array list to store childes
        ArrayList<int[]> childes = new ArrayList<>();
        //get the first and second parents
        int[] p1 = arr1.get(0).clone();
        int[] p2 = arr1.get(1).clone();
        Set<Integer> s = new HashSet<>();
        //creat childes arrays
        int[] c1 = p1.clone();
        int[] c2 = p2.clone();

        //left and right variables which represent the part where the cross over should happened
        Random random = new Random();
        int l = random.ints(0, arr.size()).findFirst().getAsInt();
        int r = random.ints(l, arr.size()).findFirst().getAsInt();
        while (l == r) {
            l = random.ints(0, arr.size()).findFirst().getAsInt();
            r = random.ints(l, arr.size()).findFirst().getAsInt();
        }
        //if left is bigger than right swap them
        if (r < l) {
            int temp;
            temp = l;
            l = r;
            r = temp;
        }
        //from left to right swap
        for (int i = l; i <= r; i++) {
            int temp;
            temp = c1[i];
            c1[i] = c2[i];
            c2[i] = temp;
        }
        //mark the dublicated projects as -1
        for (int i = 0; i < l; i++) {
            for (int j = l; j <= r; j++) {
                if (c1[i] == c1[j])
                    c1[j] = -1;
            }
        }
        for (int i = r + 1; i < c1.length; i++) {
            for (int j = l; j <= r; j++) {
                if (c1[i] == c1[j])
                    c1[j] = -1;
            }
        }
        for (int i = 0; i < l; i++) {
            for (int j = l; j <= r; j++) {
                if (c2[i] == c2[j])
                    c2[j] = -1;
            }
        }
        for (int i = r + 1; i < c2.length; i++) {
            for (int j = l; j <= r; j++) {
                if (c2[i] == c2[j])
                    c2[j] = -1;
            }
        }

        //creat set from all projects in the two parents
        for (int i = 0; i < arr.size(); i++) {
            s.add(p1[i]);
            s.add(p2[i]);
        }

        //convert the list to two array lists
        ArrayList<Integer> temp1 = new ArrayList<>(s);
        ArrayList<Integer> temp2 = new ArrayList<>(s);

        //creat two hash maps
        HashMap<Integer, Boolean> p11 = new HashMap<>();
        HashMap<Integer, Boolean> p22 = new HashMap<>();
        //fill the hashmap with  false for all the values of the parents
        for (int i = 0; i < s.size(); i++) {
            p11.put(temp1.get(i), false);
            p22.put(temp2.get(i), false);
        }
        //mark the value from childes which does not equal -1 to true in the hashmaps
        for (int i = 0; i < arr.size(); i++) {
            if (c1[i] != -1) {

                    p11.put(c1[i], true);

            }
            if (c2[i] != -1) {

                    p22.put(c2[i], true);

            }

        }

        for (int i = l; i <= r; i++) {
            if (c2[i] == -1) {
                for (Map.Entry<Integer, Boolean> u : p22.entrySet()) {
                    if (!u.getValue()) {
                        c2[i] = u.getKey();
                        p22.put(c2[i], true);
                        break;
                    }

                }
            }
        }
        for (int i = l; i <= r; i++) {
            if (c1[i] == -1) {
                for (Map.Entry<Integer, Boolean> u : p11.entrySet()) {
                    if (!u.getValue()) {
                        c1[i] = u.getKey();
                        p11.put(c1[i], true);
                        break;
                    }

                }
            }
        }
        HashMap<Integer, Integer> h = new HashMap<>();
        for (int i = 0; i < arr.size(); i++) {
            if (h.containsKey(c1[i]))
                h.put(c1[i], h.get(c1[i]) + 1);
            else
                h.put(c1[i], 1);


        }
        /*HashMap<Integer,Integer>h2 = new HashMap<>() ;
        for (int i = 0 ; i < arr.size() ; i++){
            if (h2.containsKey(c2[i]))
                h2.put(c2[i] , h2.get(c2[i])+1) ;
            else
                h2.put(c2[i] , 1) ;

        }
        System.out.println(h +"\n"+ h2);*/

        childes.add(c1);
        childes.add(c2);

        return childes;
    }

    //mutation function
    public void mutation() {
        ArrayList<int[]> result = new ArrayList<>();
        Random random = new Random();
        //generate two random numbers for chose two different indexes to swap it
        int l = random.ints(0, arr.size()).findFirst().getAsInt();
        int r = random.ints(0, arr.size()).findFirst().getAsInt();
        //creat child 1 and do the mutation for it

        int temp = arr1.get(arr1.size() - 1)[l];
        arr1.get(arr1.size() - 1)[l] = arr1.get(arr1.size() - 1)[r];
        arr1.get(arr1.size() - 1)[r] = temp;
        //do the same thing for child 2
        l = random.ints(0, arr.size()).findFirst().getAsInt();
        r = random.ints(0, arr.size()).findFirst().getAsInt();
        int[] c2 = arr1.get(arr1.size() - 2);
        temp = arr1.get(arr1.size() - 2)[l];
        arr1.get(arr1.size() - 2)[l] = arr1.get(arr1.size() - 2)[r];
        arr1.get(arr1.size() - 2)[r] = temp;


    }


    public ArrayList<int[]> getArr1() {
        return arr1;
    }

    public void setArr1(ArrayList<int[]> arr1) {
        this.arr1 = arr1;
    }

    public ArrayList<info> getArr() {
        return arr;
    }

    public void setArr(ArrayList<info> arr) {
        this.arr = arr;
    }
}
