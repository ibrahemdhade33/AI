package sample;

public class info {
    private String group ;
    private int f ;
    private int s ;
    private int t ;
    public info (String group , int f , int s , int t){
        this.group=group ;
        this.f=f ;
        this.s=s;
        this.t=t ;
    }

    public String getGroup() {
        return group;
    }

    public void setGroup(String group) {
        this.group = group;
    }

    public int getF() {
        return f;
    }

    public void setF(int f) {
        this.f = f;
    }

    public int getS() {
        return s;
    }

    public void setS(int s) {
        this.s = s;
    }

    public int getT() {
        return t;
    }

    public void setT(int t) {
        this.t = t;
    }

    @Override
    public String toString() {
        return group + "," + f + "," + s +"," + t ;
    }
}
