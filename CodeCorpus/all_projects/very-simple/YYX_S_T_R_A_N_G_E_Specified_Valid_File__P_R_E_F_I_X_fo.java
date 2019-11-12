package test;

public class fo {

	public int countSpaces(String s) {
		int n = 0;
		while (s.charAt(n) == ' ') {
			n++;
		}
		return n;
	}

}