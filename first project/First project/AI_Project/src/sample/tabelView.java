package sample;

public class tabelView {
    private int number;
    private String groups;
    private int first;
    private int second;
    private int third;
    private int alg;

    public tabelView(int number, String groups, int first, int second, int third, int alg) {
        this.number = number;
        this.groups = groups;
        this.first = first;
        this.second = second;
        this.third = third;
        this.alg = alg;
    }

    public int getNumber() {
        return number;
    }

    public void setNumber(int number) {
        this.number = number;
    }

    public String getGroups() {
        return groups;
    }

    public void setGroups(String groups) {
        this.groups = groups;
    }

    public int getFirst() {
        return first;
    }

    public void setFirst(int first) {
        this.first = first;
    }

    public int getSecond() {
        return second;
    }

    public void setSecond(int second) {
        this.second = second;
    }

    public int getThird() {
        return third;
    }

    public void setThird(int third) {
        this.third = third;
    }

    public int getAlg() {
        return alg;
    }

    public void setAlg(int alg) {
        this.alg = alg;
    }

    @Override
    public String toString() {
        return "tabelView{" +
                "groupNumber=" + number +
                ", groups='" + groups + '\'' +
                ", firstChoice=" + first +
                ", secondChoice=" + second +
                ", thirdChoice=" + third +
                ", algChoice=" + alg +
                '}';
    }

}
