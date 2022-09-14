package test;

public class TestReg {
	
	public static void main(String[] args) {
		String s = "D:/pom.xml";
		String reg = ".+[^(\\.java)]$";
		System.out.println(s.matches(reg));
		TestReg.test(0, 0);
	}
	
	public static void test(int a, int b) {
		if (a < b) {
			System.out.println("error!");
		}
	}
	
}
