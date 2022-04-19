package sample;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

//define the class for initial solution
public class initialsol {
    private final ArrayList<Integer> SelectedProjects;
    private final ArrayList<Integer> Allprojects;
    private final int size;

    public initialsol(ArrayList<Integer> SelectedProjects, ArrayList<Integer> Allprojects, int size) {
        this.SelectedProjects = SelectedProjects;
        this.Allprojects = Allprojects;
        this.size = size;
    }

    //function to generate initial solution
    public int[] sol() {
        Random r = new Random();
        //define the chromosome
        int[] d = new int[size + 1];
        //define visited array
        boolean[] vis = new boolean[Allprojects.size() + 1];

        Arrays.fill(vis, false);
        Arrays.fill(d, 0);
        int y = SelectedProjects.size();
        //file the first part of the solution []d with projects that chosen by all groups
        for (int i = 0; i < SelectedProjects.size(); i++) {
            //chose random index from the selected projects
            int x = r.ints(0, y).findFirst().getAsInt();
            //check if that index is visited
            if (!vis[SelectedProjects.get(x)])
                d[i] = SelectedProjects.get(x);

            else {
                //if visited find un visited one
                while (vis[SelectedProjects.get(x)])
                    x = r.ints(0, y).findFirst().getAsInt();

                d[i] = SelectedProjects.get(x);

            }
            //mark the chosen project as visited
            vis[SelectedProjects.get(x)] = true;
        }
        int l = Allprojects.size();
        //do the same thing for the other part of the array d
        for (int i = SelectedProjects.size(); i < Allprojects.size(); i++) {

            int x = r.ints(0, l).findFirst().getAsInt();
            if (!vis[Allprojects.get(x)])
                d[i] = Allprojects.get(x);

            else {
                while (vis[Allprojects.get(x)])
                    x = r.ints(0, l).findFirst().getAsInt();

                d[i] = Allprojects.get(x);

            }

            vis[Allprojects.get(x)] = true;
        }

        return d;
    }




}