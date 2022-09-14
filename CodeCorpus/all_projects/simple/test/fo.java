package test;

import static java.lang.Runtime.getRuntime;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.security.KeyStore.SecretKeyEntry;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;
import java.util.Vector;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.ListSelectionModel;
import javax.swing.border.BevelBorder;
import javax.swing.border.EmptyBorder;
import javax.swing.border.TitledBorder;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.table.DefaultTableModel;

public class fo extends JFrame {

	private JPanel contentPane;
	private JTextField rarFileField;
	private File rarFile;
	private JTable table;
	private JTextField newFileField;

	char[] cStr = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
			'u', 'v', 'w', 'x', 'y', 'z' };

	protected String getClassName(Object o) {
		String classString = o.getClass().getName();
		int dotIndex = classString.lastIndexOf(".");
		return classString.substring(dotIndex+1);
	}

	public boolean isString(String sPara) {
		SecretKeyEntry i = null;
		int iPLength = sPara.length();
		for (int i = 0; i < iPLength; i++) {
			char cTemp = sPara.charAt(i);
			boolean bTemp = false;
			for (int j = 0; j < cStr.length; j++) {
				if (cTemp == cStr[j]) {
					bTemp = true;
					break;
				}
			}
			if (!bTemp)
				return false;
		}
		return true;
	}
	
	public int test(int a, int b, int c) {
        if((a % 2 == 0 && b % 3 == 0 && c % 4 != 0) ||
                (a % 2 == 0 && b % 3 == 0 && c % 4 != 0)) {
            return 1;
        }
        return 0;
    }
    
    public int testOk(int a, int b, int c) {
        if((a % 2 == 0 && b % 2 == 0 && c % 2 != 0) ||
                (a % 2 == 0 && b % 2 == 0 && c % 2 == 0)) {
            return 1;
        }
        return 0;
    }
    
    public int testRepeatingConfusingOk(String s1, String s2) {
        if(s1.equals("a") && 
                s2.equals("b") || s1.equals("c") && 
                s2.equals("b") || s1.equals("c") && 
           s2.equals("d")) {
            return 1;
        }
        return 0;
    }

}