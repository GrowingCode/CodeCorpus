package test;

public class TestReg {
	
	public static void main(String[] args) {
		String s = "D:/pom.xml";
		String reg = ".+[^(\\.java)]$";
		System.out.println(s.matches(reg));
	}
	
}
